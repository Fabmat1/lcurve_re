#include "comp_light_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int FACE_BLOCK = 1024;
constexpr int PHASE_THREADS = 128;
constexpr std::size_t DEFAULT_MIN_WORK = 250000;

template<typename Real>
struct LdcCuda {
    Real c1, c2, c3, c4, base, mucrit;
};

template<typename Real>
struct DeviceGridView {
    const Real *dx, *dy, *dz;
    const Real *px, *py, *pz;
    const Real *flux;
    const Real *in1, *out1;
    const int *moff;
    const Real *min_, *mout_;
    int n, n0, n1;
};

template<typename Real>
struct DevicePhases {
    const Real *ex, *ey, *ez, *phin;
    int n;
};

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }

    bool reserve(std::size_t n) {
        if (n <= capacity_) return true;
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
        if (n == 0) return true;
        if (cudaMalloc(reinterpret_cast<void **>(&ptr_), n * sizeof(T)) !=
            cudaSuccess)
            return false;
        capacity_ = n;
        return true;
    }

    bool copy_from(const T *src, std::size_t n) {
        if (!reserve(n)) return false;
        return n == 0 || cudaMemcpy(ptr_, src, n * sizeof(T),
                                    cudaMemcpyHostToDevice) == cudaSuccess;
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

template<typename Real>
struct DeviceGrid {
    const Lcurve::FlatGrid *owner = nullptr;
    std::uint64_t generation = 0;
    int n = 0, n0 = 0, n1 = 0;

    DeviceBuffer<Real> dx, dy, dz, px, py, pz, flux;
    DeviceBuffer<Real> in1, out1, min_, mout_;
    DeviceBuffer<int> moff;
    std::vector<Real> staging;

    bool copy(DeviceBuffer<Real> &destination,
              const std::vector<double> &source) {
        if constexpr (std::is_same_v<Real, double>)
            return destination.copy_from(source.data(), source.size());
        staging.resize(source.size());
        std::transform(source.begin(), source.end(), staging.begin(),
                       [](double value) { return static_cast<Real>(value); });
        return destination.copy_from(staging.data(), staging.size());
    }

    bool upload(const Lcurve::FlatGrid &g) {
        if (!copy(dx, g.dx) || !copy(dy, g.dy) || !copy(dz, g.dz) ||
            !copy(px, g.px) || !copy(py, g.py) || !copy(pz, g.pz) ||
            !copy(flux, g.flux) || !copy(in1, g.in1) ||
            !copy(out1, g.out1) ||
            !moff.copy_from(g.moff.data(), g.moff.size()) ||
            !copy(min_, g.min_) || !copy(mout_, g.mout_))
            return false;

        n = static_cast<int>(g.n);
        n0 = static_cast<int>(g.n0);
        n1 = static_cast<int>(g.n1);
        owner = &g;
        generation = g.generation;
        return true;
    }

    DeviceGridView<Real> view() const {
        return {dx.get(), dy.get(), dz.get(), px.get(), py.get(), pz.get(),
                flux.get(), in1.get(), out1.get(), moff.get(), min_.get(),
                mout_.get(), n, n0, n1};
    }
};

template<typename Real>
__device__ __forceinline__ bool eclipsed(Real phin, Real in, Real out) {
    return (phin >= in && phin <= out) || phin <= out - Real(1);
}

template<typename Real, int LTYPE>
__device__ __forceinline__ Real intensity(Real mu, const LdcCuda<Real> &L) {
    Real m = fmin(mu, Real(1));
    Real im;
    if constexpr (LTYPE == 0) {
        Real om = Real(1) - m;
        im = Real(1) - om * (L.c1 + om * (L.c2 + om *
                                           (L.c3 + om * L.c4)));
    } else {
        Real msq = sqrt(fmax(m, Real(0)));
        im = L.base + msq * (L.c1 + msq * (L.c2 + msq *
                                           (L.c3 + msq * L.c4)));
    }
    return static_cast<Real>(mu > Real(0)) * im;
}

template<typename Real, int LTYPE, bool BEAM>
__device__ __forceinline__ Real star1_element(
    const DeviceGridView<Real> &g, int i, Real ex, Real ey, Real ez,
    const LdcCuda<Real> &L, Real beam, Real spin, Real vfac, Real xcofm) {
    Real mu = ex * g.dx[i] + ey * g.dy[i] + ez * g.dz[i];
    Real val;
    if constexpr (!BEAM) {
        val = mu * g.flux[i] * intensity<Real, LTYPE>(mu, L);
    } else {
        Real vx = -vfac * spin * g.py[i];
        Real vy = vfac * (spin * g.px[i] - xcofm);
        Real vr = -(ex * vx + ey * vy);
        Real vn = g.dx[i] * vx + g.dy[i] * vy;
        Real mud = mu - mu * vr - vn;
        val = mu * g.flux[i] * (Real(1) - beam * vr) *
              intensity<Real, LTYPE>(mud, L);
    }
    return static_cast<Real>(mu > L.mucrit) * val;
}

template<typename Real, int LTYPE, bool BEAM, bool GLENS>
__device__ __forceinline__ Real star2_element(
    const DeviceGridView<Real> &g, int i, Real ex, Real ey, Real ez,
    const LdcCuda<Real> &L, Real beam, Real spin, Real vfac, Real xcofm,
    Real rlens1) {
    Real mu = ex * g.dx[i] + ey * g.dy[i] + ez * g.dz[i];
    Real magn = Real(1);
    if constexpr (GLENS) {
        Real sx = g.px[i], sy = g.py[i], sz = g.pz[i];
        Real d = -(sx * ex + sy * ey + sz * ez);
        Real qx = sx + d * ex, qy = sy + d * ey, qz = sz + d * ez;
        Real p = sqrt(qx * qx + qy * qy + qz * qz);
        Real ph = Real(0.5) * p;
        Real rd = rlens1 * d;
        Real pd = (ph * ph > Real(25) * rd) ? p + rd / p
                                          : ph + sqrt(ph * ph + rd);
        Real mg = pd * pd / (pd - ph) / ph / Real(4);
        magn = d > Real(0) ? mg : Real(1);
    }

    Real val;
    if constexpr (!BEAM) {
        val = mu * magn * g.flux[i] * intensity<Real, LTYPE>(mu, L);
    } else {
        Real vx = -vfac * spin * g.py[i];
        Real vy = vfac * (spin * (g.px[i] - Real(1)) + Real(1) - xcofm);
        Real vr = -(ex * vx + ey * vy);
        Real vn = g.dx[i] * vx + g.dy[i] * vy;
        Real mud = mu - mu * vr - vn;
        val = mu * magn * g.flux[i] * (Real(1) - beam * vr) *
              intensity<Real, LTYPE>(mud, L);
    }
    if constexpr (GLENS)
        return mu > L.mucrit ? val : Real(0);
    return static_cast<Real>(mu > L.mucrit) * val;
}

template<typename Real, int LTYPE, bool BEAM, bool STAR2, bool GLENS>
__global__ void flux_partials(DeviceGridView<Real> g,
                              DevicePhases<Real> phases, LdcCuda<Real> L,
                              Real beam, Real spin, Real vfac, Real xcofm,
                              Real rlens1,
                              double *partials) {
    const int face_block = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k >= phases.n) return;

    const int lo = face_block * FACE_BLOCK;
    const int hi = min(g.n, lo + FACE_BLOCK);
    const Real ex = phases.ex[k], ey = phases.ey[k], ez = phases.ez[k];
    const Real phin = phases.phin[k];
    double sum = 0.0;

    // One thread owns one phase and one 1024-face block. Adjacent CUDA
    // lanes own adjacent phases, so the face data are broadcast through a
    // warp while every lane preserves the CPU kernel's ascending summation
    // order. This avoids atomics and keeps the result deterministic.
    for (int i = lo; i < hi; ++i) {
        bool hidden = false;
        if (i >= g.n0 && i < g.n1) {
            hidden = eclipsed<Real>(phin, g.in1[i - g.n0],
                                    g.out1[i - g.n0]);
        } else if (i >= g.n1) {
            const int j0 = g.moff[i - g.n1];
            const int j1 = g.moff[i - g.n1 + 1];
            for (int j = j0; j < j1; ++j) {
                if (eclipsed<Real>(phin, g.min_[j], g.mout_[j])) {
                    hidden = true;
                    break;
                }
            }
        }
        if (!hidden) {
            if constexpr (STAR2)
                sum += static_cast<double>(star2_element<Real, LTYPE, BEAM, GLENS>(
                    g, i, ex, ey, ez, L, beam, spin, vfac, xcofm, rlens1));
            else
                sum += static_cast<double>(star1_element<Real, LTYPE, BEAM>(
                    g, i, ex, ey, ez, L, beam, spin, vfac, xcofm));
        }
    }
    partials[static_cast<std::size_t>(face_block) * phases.n + k] = sum;
}

template<typename Real>
class CudaFluxContext {
public:
    ~CudaFluxContext() {
        if (profile_ && calls_)
            std::fprintf(stderr,
                         "LCURVE CUDA %s profile: %zu flux calls, grid %.3f s, "
                         "phases %.3f s, kernel+download %.3f s, merge %.3f s\n",
                         std::is_same_v<Real, double> ? "FP64" : "mixed",
                         calls_, grid_seconds_, phase_seconds_,
                         kernel_seconds_, merge_seconds_);
    }

    bool usable(std::size_t work) {
        initialise();
        return enabled_ && work >= min_work_;
    }

    DeviceGrid<Real> *grid(const Lcurve::FlatGrid &g) {
        for (auto &slot : grids_) {
            if (slot.owner == &g) {
                if (slot.generation != g.generation && !slot.upload(g)) {
                    disable("uploading a stellar grid");
                    return nullptr;
                }
                return &slot;
            }
        }

        DeviceGrid<Real> &slot = grids_[next_grid_++ % grids_.size()];
        if (!slot.upload(g)) {
            disable("uploading a stellar grid");
            return nullptr;
        }
        return &slot;
    }

    bool upload_phases(const Lcurve::PhaseBatch &p) {
        const std::size_t n = p.size();
        bool unchanged = phase_count_ == n && phase_staging_.size() == 4 * n;
        if (unchanged) {
            for (std::size_t k = 0; k < n; ++k) {
                if (phase_staging_[k] != static_cast<Real>(p.ex[k]) ||
                    phase_staging_[n + k] != static_cast<Real>(p.ey[k]) ||
                    phase_staging_[2 * n + k] != static_cast<Real>(p.ez[k]) ||
                    phase_staging_[3 * n + k] !=
                        static_cast<Real>(p.phin[k])) {
                    unchanged = false;
                    break;
                }
            }
        }
        if (unchanged) return true;

        phase_staging_.resize(4 * n);
        for (std::size_t k = 0; k < n; ++k) {
            phase_staging_[k] = static_cast<Real>(p.ex[k]);
            phase_staging_[n + k] = static_cast<Real>(p.ey[k]);
            phase_staging_[2 * n + k] = static_cast<Real>(p.ez[k]);
            phase_staging_[3 * n + k] = static_cast<Real>(p.phin[k]);
        }
        if (!phase_data_.copy_from(phase_staging_.data(),
                                   phase_staging_.size())) {
            disable("uploading phase data");
            return false;
        }
        phase_count_ = n;
        return true;
    }

    DevicePhases<Real> phases(int n) const {
        const Real *base = phase_data_.get();
        return {base, base + n, base + 2 * n, base + 3 * n, n};
    }

    bool prepare_partials(std::size_t n) {
        host_partials_.resize(n);
        if (!partials_.reserve(n)) {
            disable("allocating flux partials");
            return false;
        }
        return true;
    }

    double *device_partials() { return partials_.get(); }

    bool download_partials(std::size_t n) {
        cudaError_t err = cudaMemcpy(host_partials_.data(), partials_.get(),
                                     n * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            disable("downloading flux partials", err);
            return false;
        }
        return true;
    }

    const double *host_partials() const { return host_partials_.data(); }

    bool launch_ok() {
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return true;
        disable("launching the stellar-flux kernel", err);
        return false;
    }

    void record(double grid, double phases, double kernel, double merge) {
        ++calls_;
        if (!profile_) return;
        grid_seconds_ += grid;
        phase_seconds_ += phases;
        kernel_seconds_ += kernel;
        merge_seconds_ += merge;
    }

    std::uint64_t calls() const { return calls_; }

private:
    void initialise() {
        if (initialised_) return;
        initialised_ = true;

        std::string mode;
        if (const char *env = std::getenv("LCURVE_CUDA")) mode = env;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (mode == "0" || mode == "off" || mode == "false" ||
            mode == "cpu")
            return;
        // CUDA changes the reduction precision and has a one-time context
        // cost, so keep it opt-in. Enabling the whole-grid path implicitly
        // enables the matching flux path as well.
        if (mode.empty()) {
            const char *grid = std::getenv("LCURVE_CUDA_GRID");
            std::string grid_mode = grid ? grid : "";
            std::transform(grid_mode.begin(), grid_mode.end(),
                           grid_mode.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (grid_mode.empty() || grid_mode == "0" ||
                grid_mode == "off" || grid_mode == "false")
                return;
        }
        forced_ = mode == "1" || mode == "on" || mode == "true" ||
                  mode == "force";
        verbose_ = forced_ || std::getenv("LCURVE_CUDA_VERBOSE") != nullptr;
        profile_ = std::getenv("LCURVE_CUDA_PROFILE") != nullptr;

        if (const char *env = std::getenv("LCURVE_CUDA_MIN_WORK")) {
            char *end = nullptr;
            unsigned long long value = std::strtoull(env, &end, 10);
            if (end != env) min_work_ = static_cast<std::size_t>(value);
        }

        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) {
            if (verbose_)
                std::cerr << "LCURVE CUDA unavailable: "
                          << (err == cudaSuccess ? "no CUDA device"
                                                : cudaGetErrorString(err))
                          << "; using CPU kernels\n";
            cudaGetLastError();
            return;
        }

        int device = 0;
        if (const char *env = std::getenv("LCURVE_CUDA_DEVICE"))
            device = std::max(0, std::atoi(env));
        if (device >= count) device = 0;
        err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            if (verbose_)
                std::cerr << "LCURVE CUDA could not select device: "
                          << cudaGetErrorString(err) << "; using CPU kernels\n";
            cudaGetLastError();
            return;
        }

        enabled_ = true;
        if (verbose_) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
                std::cerr << "LCURVE CUDA flux kernels enabled on "
                          << prop.name << "\n";
        }
    }

    void disable(const char *operation, cudaError_t err = cudaGetLastError()) {
        if (verbose_ || forced_)
            std::cerr << "LCURVE CUDA failed while " << operation << ": "
                      << cudaGetErrorString(err) << "; using CPU kernels\n";
        enabled_ = false;
        cudaGetLastError();
    }

    bool initialised_ = false;
    bool enabled_ = false;
    bool forced_ = false;
    bool verbose_ = false;
    bool profile_ = false;
    std::size_t min_work_ = DEFAULT_MIN_WORK;
    std::array<DeviceGrid<Real>, 4> grids_;
    std::size_t next_grid_ = 0;
    DeviceBuffer<Real> phase_data_;
    DeviceBuffer<double> partials_;
    std::vector<Real> phase_staging_;
    std::size_t phase_count_ = 0;
    std::vector<double> host_partials_;
    std::size_t calls_ = 0;
    double grid_seconds_ = 0.0, phase_seconds_ = 0.0;
    double kernel_seconds_ = 0.0, merge_seconds_ = 0.0;
};

template<typename Real>
CudaFluxContext<Real> &context() {
    static thread_local CudaFluxContext<Real> ctx;
    return ctx;
}

template<typename Real, int LTYPE, bool BEAM, bool STAR2, bool GLENS>
bool run_flux(const Lcurve::FlatGrid &grid,
              const Lcurve::PhaseBatch &phases, const LdcCuda<Real> &ldc,
              double beam, double spin, double vfac, double xcofm,
              double rlens1, double *out) {
    const int nphase = static_cast<int>(phases.size());
    const int nblocks = static_cast<int>((grid.n + FACE_BLOCK - 1) /
                                         FACE_BLOCK);
    if (nphase == 0 || nblocks == 0) return false;

    CudaFluxContext<Real> &ctx = context<Real>();
    if (!ctx.usable(grid.n * phases.size())) return false;
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    DeviceGrid<Real> *device_grid = ctx.grid(grid);
    auto t1 = Clock::now();
    if (!device_grid || !ctx.upload_phases(phases)) return false;
    auto t2 = Clock::now();

    const std::size_t npartials = static_cast<std::size_t>(nblocks) * nphase;
    if (!ctx.prepare_partials(npartials)) return false;

    dim3 block(PHASE_THREADS);
    dim3 launch_grid(nblocks, (nphase + PHASE_THREADS - 1) / PHASE_THREADS);
    flux_partials<Real, LTYPE, BEAM, STAR2, GLENS><<<launch_grid, block>>>(
        device_grid->view(), ctx.phases(nphase), ldc,
        static_cast<Real>(beam), static_cast<Real>(spin),
        static_cast<Real>(vfac), static_cast<Real>(xcofm),
        static_cast<Real>(rlens1), ctx.device_partials());
    if (!ctx.launch_ok() || !ctx.download_partials(npartials)) return false;
    auto t3 = Clock::now();

    std::fill(out, out + nphase, 0.0);
    const double *partials = ctx.host_partials();
    for (int b = 0; b < nblocks; ++b)
        for (int k = 0; k < nphase; ++k)
            out[k] += partials[static_cast<std::size_t>(b) * nphase + k];
    auto t4 = Clock::now();
    const auto seconds = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    ctx.record(seconds(t0, t1), seconds(t1, t2), seconds(t2, t3),
               seconds(t3, t4));
    return true;
}

template<typename Real>
LdcCuda<Real> make_ldc(const Lcurve::LDC &ldc) {
    return {static_cast<Real>(ldc.c1()), static_cast<Real>(ldc.c2()),
            static_cast<Real>(ldc.c3()), static_cast<Real>(ldc.c4()),
            static_cast<Real>(1.0 - (ldc.c1() + ldc.c2() + ldc.c3() +
                                     ldc.c4())),
            static_cast<Real>(ldc.mucrit_val())};
}

bool use_fp64() {
    static const bool value = [] {
        const char *env = std::getenv("LCURVE_CUDA_PRECISION");
        if (!env) return false;
        std::string mode(env);
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return mode == "double" || mode == "fp64" || mode == "64";
    }();
    return value;
}

template<typename Real>
bool dispatch_star1(const Lcurve::FlatGrid &grid,
                    const Lcurve::PhaseBatch &phases, const Lcurve::LDC &ldc,
                    double beam, double spin, double vfac, double xcofm,
                    double *out) {
    const LdcCuda<Real> L = make_ldc<Real>(ldc);
    const bool claret = ldc.type() == Lcurve::LDC::CLARET;
    const bool beaming = beam != 0.0;
    if (claret)
        return beaming
            ? run_flux<Real, 1, true, false, false>(
                  grid, phases, L, beam, spin, vfac, xcofm, 0.0, out)
            : run_flux<Real, 1, false, false, false>(
                  grid, phases, L, beam, spin, vfac, xcofm, 0.0, out);
    return beaming
        ? run_flux<Real, 0, true, false, false>(
              grid, phases, L, beam, spin, vfac, xcofm, 0.0, out)
        : run_flux<Real, 0, false, false, false>(
              grid, phases, L, beam, spin, vfac, xcofm, 0.0, out);
}

template<typename Real>
bool dispatch_star2(const Lcurve::FlatGrid &grid,
                    const Lcurve::PhaseBatch &phases, const Lcurve::LDC &ldc,
                    double beam, double spin, double vfac, double xcofm,
                    bool glens1, double rlens1, double *out) {
    const LdcCuda<Real> L = make_ldc<Real>(ldc);
    const bool claret = ldc.type() == Lcurve::LDC::CLARET;
    const bool beaming = beam != 0.0;
    if (glens1) {
        if (claret)
            return beaming
                ? run_flux<Real, 1, true, true, true>(
                      grid, phases, L, beam, spin, vfac, xcofm, rlens1, out)
                : run_flux<Real, 1, false, true, true>(
                      grid, phases, L, beam, spin, vfac, xcofm, rlens1, out);
        return beaming
            ? run_flux<Real, 0, true, true, true>(
                  grid, phases, L, beam, spin, vfac, xcofm, rlens1, out)
            : run_flux<Real, 0, false, true, true>(
                  grid, phases, L, beam, spin, vfac, xcofm, rlens1, out);
    }
    if (claret)
        return beaming
            ? run_flux<Real, 1, true, true, false>(
                  grid, phases, L, beam, spin, vfac, xcofm, rlens1, out)
            : run_flux<Real, 1, false, true, false>(
                  grid, phases, L, beam, spin, vfac, xcofm, rlens1, out);
    return beaming
        ? run_flux<Real, 0, true, true, false>(
              grid, phases, L, beam, spin, vfac, xcofm, rlens1, out)
        : run_flux<Real, 0, false, true, false>(
              grid, phases, L, beam, spin, vfac, xcofm, rlens1, out);
}

} // unnamed namespace

bool Lcurve::cuda_sum_star1_multi(const FlatGrid &grid,
                                  const PhaseBatch &phases, const LDC &ldc,
                                  double beam, double spin, double vfac,
                                  double xcofm, double *out) {
    return use_fp64()
        ? dispatch_star1<double>(grid, phases, ldc, beam, spin, vfac,
                                 xcofm, out)
        : dispatch_star1<float>(grid, phases, ldc, beam, spin, vfac,
                                xcofm, out);
}

bool Lcurve::cuda_sum_star2_multi(const FlatGrid &grid,
                                  const PhaseBatch &phases, const LDC &ldc,
                                  double beam, double spin, double vfac,
                                  double xcofm, bool glens1, double rlens1,
                                  double *out) {
    return use_fp64()
        ? dispatch_star2<double>(grid, phases, ldc, beam, spin, vfac,
                                 xcofm, glens1, rlens1, out)
        : dispatch_star2<float>(grid, phases, ldc, beam, spin, vfac,
                                xcofm, glens1, rlens1, out);
}

std::uint64_t Lcurve::cuda_flux_evaluation_count() {
    return context<float>().calls() + context<double>().calls();
}
