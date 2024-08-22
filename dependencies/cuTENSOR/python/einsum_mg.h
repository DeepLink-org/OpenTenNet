#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
#include <array>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>
#include <c10/core/TensorOptions.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cutensorMg.h"


bool CHECK_MG_success(cudaError_t status) {
    return status == cudaSuccess;
}

const char* CHECK_MG_pretty(cudaError_t status) {
    return cudaGetErrorName(status);
}

bool CHECK_MG_success(cutensorStatus_t status) {
    return status == CUTENSOR_STATUS_SUCCESS;
}

const char* CHECK_MG_pretty(cutensorStatus_t status) {
    return cutensorGetErrorString(status);
}

template<typename T>
T product(const std::vector<T> &values) {
    T result = 1;
    for (auto& value : values) {
        result *= value;
    }
    return result;
}

template<typename T, typename U>
std::vector<T> multiply(const std::vector<T> &lhs, const std::vector<U> &rhs) {
    std::vector<T> result;
    assert(lhs.size() == rhs.size() || lhs.empty() || rhs.empty());
    for (size_t i = 0; i < lhs.size(); i++) {
        result.push_back((lhs.empty() ? 1 : lhs[i]) * (rhs.empty() ? 1 : rhs[i]));
    }
    return result;
}

template<typename T, typename U>
std::vector<T> discretize(const std::vector<T> &in, const std::vector<U> &block) {
    if (in.empty()) {
        return in;
    }
    if (block.empty()) {
        return in;
    }

    std::vector<T> result;
    assert(in.size() == block.size());
    for (size_t i = 0; i < in.size(); i++) {
        U b = block[i];
        result.push_back(b * ((in[i] + b - 1) / b));
    }
    return result;
}

#define CHECK_MG(x) do { auto CHECK_MG_err = (x); if (! CHECK_MG_success(CHECK_MG_err)) { \
  printf("Error (%s:%d): \"%s\" returned %s (%d)\n", __FILE__, __LINE__, \
    #x, CHECK_MG_pretty(CHECK_MG_err), CHECK_MG_err); exit(-1);} } while(0)


class CutensorMgConfig {
public:
    bool static Init(const int32_t numDevices) {
        if (numDevices <= 1) {
            throw std::runtime_error("cutensorMg error: numDevices <= 1");
        }
        devices.resize(numDevices);
        std::iota(devices.begin(), devices.end(), 0);
        return true;
    }
    static std::vector<int32_t>& GetDevices() {
        if (devices.size() <= 1) {
            throw std::runtime_error("cutensorMg error: devices.size() <= 1");
        }
        return devices;
    }
private:
    static std::vector<int32_t> devices;
};
std::vector<int32_t> CutensorMgConfig::devices = {};

cutensorMgHandle_t CreateHandle() {
    cutensorMgHandle_t handle;
    std::vector<int32_t>& devices = CutensorMgConfig::GetDevices();
    CHECK_MG(cutensorMgCreate(&handle, devices.size(), devices.data()));
    return handle;
}
cutensorMgHandle_t& GetHandle() {
    static cutensorMgHandle_t handle = CreateHandle();
    return handle;
}

class TensorMg {
public:
    const int64_t kElementSize = 8;
    TensorMg () {}
    TensorMg (const std::vector<int64_t>& shape, 
              const std::vector<int64_t>& blockSize,
              const std::vector<int32_t>& deviceCount
             ):
              numModes_(shape.size()),
              block_size_(blockSize),
              device_count_(deviceCount),
              shape_(shape) {
        if (numModes_ != block_size_.size() || numModes_ != device_count_.size()) {
            throw std::runtime_error("cutensorMg error: numModes_ != block_size_.size() || numModes_ != device_count_.size()");
        }
        for (size_t i = 0; i < numModes_; i++) {
            if (shape_[i] < block_size_[i]) {
                throw std::runtime_error("cutensorMg error: shape_[i] < block_size_[i]");
            }
        }
        std::reverse(block_size_.rbegin(), block_size_.rend());
        std::reverse(device_count_.rbegin(), device_count_.rend());
        initElements();
        initDataUseTensor();
    }

    void initElements () {
        extent_.resize(numModes_, 0);
        for (uint32_t i = 0; i < numModes_; i++) {
            extent_[i] = shape_[numModes_ - i - 1];    
        }

        const std::vector<int32_t>& devices = CutensorMgConfig::GetDevices();
        int32_t numDevices = devices.size();
        
        int32_t n = product(device_count_);
        for(int32_t i=0; i < n; ++i) {
            block_devices_.push_back(devices[i%numDevices]);
        }
        elements_ = product(discretize(extent_, multiply(device_count_, block_size_))) / product(device_count_);
    }
    void initDataUseMalloc() {
        for (auto& device : this->block_devices_) {
            void* memory;
            CHECK_MG(cudaSetDevice(device));
            CHECK_MG(cudaMalloc(&memory, elements_ * kElementSize));
            this->data_.push_back(memory);
        }
        this->needFreeMemory_ = true;
    }
    void initDataUseTensor() {
        for (auto& device : this->block_devices_) {
            at::Tensor tmp = at::zeros(elements_,
                                       at::TensorOptions(at::device({at::kCUDA, device})
                                                        .dtype(at::ScalarType::ComplexFloat)));
            this->tensors_.push_back(tmp);
            this->data_.push_back(tmp.data_ptr());
        }
        this->needFreeMemory_ = false;
    }

    // void init (const std::vector<int64_t>& shape) {
    //     this->shape_ = shape;
    //     numModes_ = shape.size();
    //     extent_.resize(numModes_, 0);
    //     block_size_.resize(numModes_, 0);
    //     device_count_.resize(numModes_, 1);
    //     size_t maxId = 0;
    //     for (uint32_t i = 0; i < numModes_; i++) {
    //         extent_[i] = shape[numModes_ - i - 1];
    //         block_size_[i] = extent_[i];
    //         if (extent_[i] > extent_[maxId]) {
    //             maxId = i;
    //         }
    //     }
    //     const std::vector<int32_t>& devices = CutensorMgConfig::GetDevices();
    //     size_t numDevices = devices.size();
    //     // block_size_[maxId] = ceil(extent_[maxId] * 0.1 / numDevices);
    //     if (numModes_ > 20) {
    //         block_size_[0] = 512;
    //         // block_size_[8] = 1;
    //         // block_size_[6] = 1;
    //         // block_size_[14] = 1;
    //     } else {
    //         block_size_[0] = 512;
    //         // block_size_[7] = 1;
    //         // block_size_[8] = 1;
    //         // block_size_[16] = 1;
    //     }

    //     remaining_devices_ = numDevices;
    //     bool changed = true;
    //     while (changed) {
    //         changed = false;
    //         for (int i = numModes_ - 1; i >= 0 && remaining_devices_ > 1; i = i - 1) {
    //             int32_t maxDeviceCount = extent_[i] / block_size_[i];
    //             if (device_count_[i] < maxDeviceCount) {
    //                 device_count_[i] *= 2;
    //                 remaining_devices_ /= 2;
	// 	            changed = true;
    //             }
    //         }
    //     }

    //     int64_t elements = 1;
    //     for (size_t i = 0; i < numModes_; i++) {
    //         int64_t numBlocks = (extent_[i] + block_size_[i] - 1) / block_size_[i];
    //         int64_t numBlocksPerDevice = (numBlocks + device_count_[i] - 1) / device_count_[i];
    //         elements *= numBlocksPerDevice * block_size_[i];
    //     }

    //     for (size_t i = 0; i < numDevices; i++) {
    //         CHECK_MG(cudaSetDevice(i));
    //         void* ptr;
    //         CHECK_MG(cudaMalloc(&ptr, elements * kElementSize));
    //         this->data_.push_back(ptr);
    //     }
    //     this->needFreeMemory_ = true;
    // }

    TensorMg (const torch::Tensor& tensor) {
        this->shape_ = tensor.sizes().vec();
        this->numModes_ = this->shape_.size();
        extent_.resize(this->numModes_, 0);
        for (uint32_t i = 0; i < this->numModes_; i++) {
            extent_[i] = shape_[this->numModes_ - i - 1];
        }
        auto device = tensor.device();
        if (device.is_cpu()) {
            this->block_devices_.push_back(CUTENSOR_MG_DEVICE_HOST);
        } else if (device.is_cuda()) {
            this->block_devices_.push_back(device.index());
        } else {
            throw std::runtime_error("cutensorMg: only support cpu and cuda."); 
        }
        this->data_.push_back(tensor.data_ptr());
        this->needFreeMemory_ = false;
    }

    ~TensorMg() {
        if (this->needFreeMemory_) {
            for (auto& ptr : this->data_) {
                if (ptr != nullptr) {
                    CHECK_MG(cudaFree(ptr));
                }
            }
        }
    }

    static bool copyMg(TensorMg& dst, cutensorMgTensorDescriptor_t& descDst,
                     TensorMg& src, cutensorMgTensorDescriptor_t& descSrc) {
        std::vector<int32_t> modesSrc(src.getNumModes());
        for (size_t i = 0; i < src.getNumModes(); i++) {
            modesSrc[i] = 'a' + i;
        }
        std::vector<int32_t> modesDst(dst.getNumModes());
        for (size_t i = 0; i < dst.getNumModes(); i++) {
            modesDst[i] = 'a' + i;
        }
        const std::vector<int32_t>& devices = CutensorMgConfig::GetDevices();
        const cutensorMgHandle_t& handle = GetHandle();
        cutensorMgCopyDescriptor_t descCopy;
        CHECK_MG(cutensorMgCreateCopyDescriptor(handle, &descCopy, descDst, modesDst.data(), descSrc, modesSrc.data()));
    
        std::vector<int64_t> deviceWorkSpaceSize(devices.size());
        int64_t hostWorkSpaceSize;
        CHECK_MG(cutensorMgCopyGetWorkspace(handle, descCopy, deviceWorkSpaceSize.data(), &hostWorkSpaceSize));

        cutensorMgCopyPlan_t plan;
        CHECK_MG(cutensorMgCreateCopyPlan(handle, &plan, descCopy, deviceWorkSpaceSize.data(), hostWorkSpaceSize));

        // host
        void* hostWorkSpace = NULL;
        CHECK_MG(cudaMallocHost(&hostWorkSpace, hostWorkSpaceSize));

        // devices
        std::vector<void*> deviceWorkSpace;
        for (size_t i = 0; i < devices.size(); i++) {
            void* memory;
            CHECK_MG(cudaSetDevice(devices[i]));
            CHECK_MG(cudaMalloc(&memory, deviceWorkSpaceSize[i]));
            deviceWorkSpace.push_back(memory);
        }

        std::vector<cudaStream_t> streams;
        for (auto& device : devices) {
            CHECK_MG(cudaSetDevice(device));
            streams.push_back(at::cuda::getCurrentCUDAStream().stream());
        }
        std::vector<const void*> ptrSrc;
        for (size_t i = 0; i < src.getData().size(); i++) {
            ptrSrc.push_back(src.getData()[i]);
        }
        CHECK_MG(cutensorMgCopy(handle, plan, dst.getData().data(), 
                    ptrSrc.data(), deviceWorkSpace.data(), 
                    hostWorkSpace, streams.data()));

        for (auto& deviceId : devices) {
            CHECK_MG(cudaSetDevice(deviceId));
            CHECK_MG(cudaDeviceSynchronize());
        }
        CHECK_MG(cudaFreeHost(hostWorkSpace));
        for (auto& p : deviceWorkSpace) {
            CHECK_MG(cudaFree(p));
        }
        CHECK_MG(cutensorMgDestroyCopyPlan(plan));
        CHECK_MG(cutensorMgDestroyCopyDescriptor(descCopy));
        return true;
    }

    template<typename ComputeType>
    static bool fromTensor(TensorMg& dst, const torch::Tensor& srcTensor) {
        TensorMg src(srcTensor);
        const cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
        const cutensorMgHandle_t& handle = GetHandle();
        cutensorMgTensorDescriptor_t descSrc;
        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descSrc, src.getNumModes(),
            src.getExtent().data(), NULL, NULL, NULL,
            NULL, src.getBlockDevices().size(), src.getBlockDevices().data(), cudaType));

        cutensorMgTensorDescriptor_t descDst;
        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descDst, dst.getNumModes(),
            dst.getExtent().data(), NULL, dst.getBlockSize().data(), NULL,
            dst.getDeviceCount().data(), dst.getBlockDevices().size(), dst.getBlockDevices().data(), cudaType));
            // dst.getDeviceCount().data(), devices.size() / dst.getRemainingDevices(), devices.data(), cudaType));

        copyMg(dst, descDst, src, descSrc);

        CHECK_MG(cutensorMgDestroyTensorDescriptor(descSrc));
        CHECK_MG(cutensorMgDestroyTensorDescriptor(descDst));
        return true;
    }

    template<typename ComputeType>
    static bool toTensor(const torch::Tensor& dstTensor, TensorMg& src) {
        // __asm__("int3");
        cutensorMgTensorDescriptor_t descSrc;
        const cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
        const cutensorMgHandle_t& handle = GetHandle();

        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descSrc, src.getNumModes(),
            src.getExtent().data(), NULL, src.getBlockSize().data(), NULL,
            src.getDeviceCount().data(), src.getBlockDevices().size(), src.getBlockDevices().data(), cudaType));
            // src.getDeviceCount().data(), devices.size() / src.getRemainingDevices(), devices.data(), cudaType));

        TensorMg dst(dstTensor);
        cutensorMgTensorDescriptor_t descDst;
        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descDst, dst.getNumModes(),
            dst.getExtent().data(), NULL, NULL, NULL,
            NULL, dst.getBlockDevices().size(), dst.getBlockDevices().data(), cudaType));
        
        copyMg(dst, descDst, src, descSrc);

        CHECK_MG(cutensorMgDestroyTensorDescriptor(descSrc));
        CHECK_MG(cutensorMgDestroyTensorDescriptor(descDst));
        return true;
    }

    std::vector<int32_t>& getBlockDevices() {
        return this->block_devices_;
    }
    std::vector<int32_t>& getDeviceCount() {
        return this->device_count_;
    }
    std::vector<int64_t>& getExtent() {
        return this->extent_;
    }
    std::vector<int64_t>& getBlockSize() {
        return this->block_size_;
    }
    std::vector<void*>& getData() {
        return this->data_;
    }
    uint32_t getNumModes() {
        return this->numModes_;
    }
    void setNumModes(uint32_t number) {
        this->numModes_ = number;
    }
    uint32_t getRemainingDevices() {
        return this->remaining_devices_;
    }
    std::vector<int64_t> getShape() {
        return this->shape_;
    }
    std::vector<at::Tensor> getTensors() {
        return this->tensors_;
    }

private:
    uint32_t numModes_;
    bool needFreeMemory_;
    std::vector<int32_t> block_devices_;
    std::vector<int64_t> extent_;
    std::vector<int64_t> block_size_;
    std::vector<int32_t> device_count_;
    std::vector<void*> data_;
    int64_t elements_;
    uint32_t remaining_devices_;
    std::vector<int64_t> shape_;
    std::vector<at::Tensor> tensors_;
};

class EinsumMg {
public:
    const static int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    EinsumMg(const std::string &equation, TensorMg& inputA, TensorMg& inputB) :
             numModesC_(0), isInitialized_(false) {
        this->numModesA_ = inputA.getNumModes();
        this->numModesB_ = inputB.getNumModes();
        this->modesA_.resize(this->numModesA_, 0);
        this->modesB_.resize(this->numModesB_, 0);
        std::vector<int64_t>& extentA = inputA.getExtent();
        std::vector<int64_t>& extentB = inputB.getExtent();
        const auto arrow_pos = equation.find("->");
        const auto comma_pos = equation.find(",");
        const auto dots = equation.find("...");
        const bool isBroadcast = (dots != std::string::npos);
        const bool isImplicit = (arrow_pos == std::string::npos);
        if (isBroadcast) {
            return;
        }
        const bool usesB = (comma_pos != std::string::npos);
        if (! usesB) {
            std::cerr << "Must use B in my einsumMg" << std::endl;
            return;
        }

        size_t a_start = 0;
        size_t a_end = isImplicit ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos) : 
                                    ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
        size_t b_start = usesB ? comma_pos + 1 : 0;
        size_t b_end   = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
        size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
        size_t c_end = equation.size();

        char modeA[kMaxNumModes_ + 2];
        uint32_t numModesA = 0;
        for (size_t i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i) {
            if (equation.at(i) != ' ') { // skip spaces
                modeA[numModesA++] = equation.at(i);
            }
        }

        char modeB[kMaxNumModes_ + 2];
        uint32_t numModesB = 0;
        for (size_t i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') { // skip spaces
                modeB[numModesB++] = equation.at(i);
            }
        }

        char modeC[kMaxNumModes_ + 2];
        uint32_t numModesC = 0;
        for (size_t i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i){
            if (equation.at(i) != ' ') { // skip spaces
                modeC[numModesC++] = equation.at(i);
            }
        }

        if ((numModesA != numModesA_) || (numModesB != numModesB_)) {
            // substring size and shape don't match
            return;
        }
        if (numModesA_ > kMaxNumModes_ || numModesB_ > kMaxNumModes_) {
            // too many modes
            return;
        }
        /**
         * Copy all modes from modeA to modeC if they don't appear in modeB
         */
        auto copyModesIf = [](const char* modeA, uint32_t numModesA,
                const char* modeB, uint32_t numModesB,
                char* modeC, uint32_t &numModesC) {
            for (uint32_t i = 0; i < numModesA; i++) {
                auto mode = modeA[i];
                bool found = false;
                for(uint32_t j=0; j < numModesB; ++j) {
                    if(mode == modeB[j]) {
                        found = true;
                        break;
                    }
                }

                if (!found) { // is non-contracted mode
                    modeC[numModesC++] = mode;
                    if (numModesC > kMaxNumModes_) {
                        // too many modes
                        return false;
                    }
                }
            }
            return true;
        };
        std::array<char, kMaxNumModes_+1> implicitModeC;
        char* redirectModeC;
        if (isImplicit) {
            // we have to copy all non-contracted modes from A over to C
            if (copyModesIf(modeA, numModesA_, modeB, numModesB_, implicitModeC.data(), numModesC_) == false) {
                return;
            }
            // we have to copy all non-contracted modes from B over to C
            if (copyModesIf(modeB, numModesB_, modeA, numModesA_, implicitModeC.data(), numModesC_) == false) {
                return;
            }
            std::sort(implicitModeC.begin(), std::next(implicitModeC.begin(), numModesC_)); // modes are sorted w.r.t. lexical order
            implicitModeC[numModesC_] = '\0';
            redirectModeC = implicitModeC.data();
        } else {
            redirectModeC = modeC;
            numModesC_ = numModesC;
        }

        for (uint32_t i = 0; i < numModesA_; i++) {
            modesA_[i] = modeA[numModesA_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesB_; i++) {
            modesB_[i] = modeB[numModesB_ - i - 1];
        }

        for (uint32_t i = 0; i < numModesC_; i++) {
            const auto mode = redirectModeC[numModesC_ - i - 1];
            modesC_[i] = mode;
            bool found = false;
            for (uint32_t j=0; j < numModesA_; ++j) {
                if (modesA_[j] == mode) {
                    extentC_[i] = extentA[j];
                    found = true;
                    break;
                }
            }
            for (uint32_t j=0; !found && j < numModesB_; ++j) {
                if (modesB_[j] == mode) {
                    extentC_[i] = extentB[j];
                    break;
                }
            }
        }
        isInitialized_ = true;
    }

    bool isInitialized() const {
        return this->isInitialized_;
    }
    std::vector<int64_t> getOutputShape() const {
        if (!this->isInitialized_) return {};
        std::vector<int64_t> extentC(numModesC_);
        for (uint32_t i=0; i < numModesC_; ++i) {
            extentC[i] = extentC_.at(numModesC_ - i - 1);
        }
        return extentC;
    }
    template<typename ComputeType>
    bool execute(TensorMg& A, TensorMg& B, TensorMg& C) const {
        if (false == this->isInitialized_) {
            return false;
        }
        const cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
        const cutensorComputeType_t computeType = CuTensorTypeTraits<ComputeType>::cutensorType;
        const std::vector<int32_t>& devices = CutensorMgConfig::GetDevices();
        const cutensorMgHandle_t& handle = GetHandle();
        cutensorMgTensorDescriptor_t descA;
        // CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descA, A.getNumModes(),
        //     A.getExtent().data(), NULL, A.getBlockSize().data(), NULL,
        //     A.getDeviceCount().data(), devices.size() / A.getRemainingDevices(), devices.data(), cudaType));

        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descA, A.getNumModes(),
            A.getExtent().data(), NULL, A.getBlockSize().data(), NULL,
            A.getDeviceCount().data(), A.getBlockDevices().size(), A.getBlockDevices().data(), cudaType));
        
        cutensorMgTensorDescriptor_t descB;
        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descB, B.getNumModes(),
            B.getExtent().data(), NULL, B.getBlockSize().data(), NULL,
            B.getDeviceCount().data(),  B.getBlockDevices().size(), B.getBlockDevices().data(), cudaType));
            // B.getDeviceCount().data(), devices.size() / B.getRemainingDevices(), devices.data(), cudaType));
        
        cutensorMgTensorDescriptor_t descC;
        CHECK_MG(cutensorMgCreateTensorDescriptor(handle, &descC, C.getNumModes(),
            C.getExtent().data(), NULL, C.getBlockSize().data(), NULL,
            C.getDeviceCount().data(),  C.getBlockDevices().size(), C.getBlockDevices().data(), cudaType));

        const cutensorWorksizePreference_t kWorksizePreference = CUTENSOR_WORKSPACE_RECOMMENDED;
        cutensorMgContractionDescriptor_t contractionDesc;
        CHECK_MG(cutensorMgCreateContractionDescriptor(handle, &contractionDesc,
                descA, modesA_.data(),
                descB, modesB_.data(),
                descC, modesC_.data(),
                descC, modesC_.data(),
                computeType));
        cutensorMgContractionFind_t contractionFind;
        CHECK_MG(cutensorMgCreateContractionFind(handle, &contractionFind,
                CUTENSORMG_ALGO_DEFAULT));

        std::vector<int64_t> workspaceSize(devices.size());
        int64_t workspaceHostSize;
        CHECK_MG(cutensorMgContractionGetWorkspace(handle,
              contractionDesc, contractionFind, kWorksizePreference, workspaceSize.data(), &workspaceHostSize));
 
        cutensorMgContractionPlan_t plan;
        CHECK_MG(cutensorMgCreateContractionPlan(handle, &plan,
              contractionDesc, contractionFind, workspaceSize.data(), workspaceHostSize));
        std::vector<cudaStream_t> streams;
        for (auto& device : devices) {
            CHECK_MG(cudaSetDevice(device));
            streams.push_back(at::cuda::getCurrentCUDAStream().stream());
        }
        /*
        * Allocate workspace
        */
        // host
        void* workspaceHost = nullptr;
        CHECK_MG(cudaMallocHost(&workspaceHost, workspaceHostSize));

        // devices
        std::vector<void*> workspaceDevice;
        for (size_t i = 0; i < devices.size(); i++) {
            void* memory;
            CHECK_MG(cudaSetDevice(devices[i]));
            CHECK_MG(cudaMalloc(&memory, workspaceSize[i]));
            workspaceDevice.push_back(memory);
        }
        int currentDeviceId = -1;
        CHECK_MG(cudaGetDevice(&currentDeviceId));
        float kAlpha = 1;
        float kBeta = 0;

        float minElapsed = 0;
        
        for (auto& deviceId : devices) {
            CHECK_MG(cudaSetDevice(deviceId));
            CHECK_MG(cudaDeviceSynchronize());
        }
        
        const auto start = std::chrono::steady_clock::now();
        CHECK_MG(cutensorMgContraction(handle, plan, &kAlpha,
                const_cast<const void**>(A.getData().data()),
                const_cast<const void**>(B.getData().data()), &kBeta, 
                const_cast<const void**>(C.getData().data()), C.getData().data(),
                workspaceDevice.data(), workspaceHost, streams.data()));
        for (auto& deviceId : devices) {
            CHECK_MG(cudaSetDevice(deviceId));
            CHECK_MG(cudaDeviceSynchronize());
        }
        const auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> dur = end - start;
        if (minElapsed == 0 || minElapsed > dur.count()) {
            minElapsed = dur.count();
        }
        
        printf("multi-gpu execution minElapsed: %.2e millisec.\n", minElapsed);

        CHECK_MG(cudaSetDevice(currentDeviceId));
        
        CHECK_MG(cudaFreeHost(workspaceHost));
        for (auto& p : workspaceDevice) {
            CHECK_MG(cudaFree(p));
        }
        CHECK_MG(cutensorMgDestroyContractionDescriptor(contractionDesc));
        CHECK_MG(cutensorMgDestroyContractionFind(contractionFind));
        CHECK_MG(cutensorMgDestroyContractionPlan(plan));
        CHECK_MG(cutensorMgDestroyTensorDescriptor(descA));
        CHECK_MG(cutensorMgDestroyTensorDescriptor(descB));
        CHECK_MG(cutensorMgDestroyTensorDescriptor(descC));
        return true;
    }
private:
    uint32_t numModesA_;
    uint32_t numModesB_;
    uint32_t numModesC_;
    std::vector<int32_t> modesA_;
    std::vector<int32_t> modesB_;
    std::array<int32_t, kMaxNumModes_> modesC_;
    std::array<int64_t, kMaxNumModes_> extentC_;
    bool isInitialized_;
};
