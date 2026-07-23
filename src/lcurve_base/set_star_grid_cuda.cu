#include "set_star_grid_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr float TWOPI = 2.0f * PI;
constexpr int THREADS = 256;

struct Vec3f {
    float x, y, z;
};

struct GridParams {
    float q, cosi, sini, r1, r2, rref1, rref2;
    float spin1, spin2, gref, pref1, pref2, delta;
    int which_star;
    int roche1, roche2, eclipse;
    double q64, cosi64, sini64, rref2_64, spin2_64, pref2_64, delta64;
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
        if (cudaMalloc(reinterpret_cast<void **>(&ptr_), n * sizeof(T)) !=
            cudaSuccess)
            return false;
        capacity_ = n;
        return true;
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

private:
    T *ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

__device__ __forceinline__ float dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3f add(Vec3f a, Vec3f b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Vec3f scale(Vec3f a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ float potential(float q, int star, float spin,
                                            Vec3f p) {
    const float mu = q / (1.0f + q);
    const float comp = 1.0f - mu;
    const float x2y2 = p.x * p.x + p.y * p.y;
    const float r1sq = x2y2 + p.z * p.z;
    const float d1 = sqrtf(r1sq);
    const float d2 = sqrtf(r1sq + 1.0f - 2.0f * p.x);
    const float ssq = spin * spin;
    if (star == static_cast<int>(Roche::PRIMARY))
        return -comp / d1 - mu / d2 - 0.5f * ssq * x2y2 + mu * p.x;
    return -comp / d1 - mu / d2
           - ssq * (0.5f + 0.5f * x2y2 - p.x) - comp * p.x;
}

__device__ __forceinline__ Vec3f potential_gradient(float q, int star,
                                                     float spin, Vec3f p) {
    const float mu = q / (1.0f + q);
    const float comp_mass = 1.0f - mu;
    const float r1sq = dot(p, p);
    const float r2sq = r1sq + 1.0f - 2.0f * p.x;
    const float d1 = sqrtf(r1sq), d2 = sqrtf(r2sq);
    const float c1 = comp_mass / (d1 * r1sq);
    const float c2 = mu / (d2 * r2sq);
    const float ssq = spin * spin;
    if (star == static_cast<int>(Roche::PRIMARY))
        return {c1 * p.x + c2 * (p.x - 1.0f) - ssq * p.x + mu,
                c1 * p.y + c2 * p.y - ssq * p.y,
                c1 * p.z + c2 * p.z};
    return {c1 * p.x + c2 * (p.x - 1.0f) - ssq * (p.x - 1.0f)
                + mu - 1.0f,
            c1 * p.y + c2 * p.y - ssq * p.y,
            c1 * p.z + c2 * p.z};
}

__device__ __forceinline__ float ray_potential(const GridParams &p,
                                                Vec3f direction, float radius) {
    const bool primary = p.which_star == static_cast<int>(Roche::PRIMARY);
    const float mu = p.q / (1.0f + p.q), comp = 1.0f - mu;
    const float spin = primary ? p.spin1 : p.spin2;
    const float cx = primary ? 0.0f : 1.0f;
    const float rdx = radius * direction.x;
    const float px = cx + rdx;
    const float py = radius * direction.y;
    const float x2y2 = px * px + py * py;
    const float companion = sqrtf(radius * radius + 1.0f +
                                  (primary ? -2.0f : 2.0f) * rdx);
    if (primary)
        return -comp / radius - mu / companion
               - 0.5f * spin * spin * x2y2 + mu * px;
    return -comp / companion - mu / radius
           - spin * spin * (0.5f + 0.5f * x2y2 - px) - comp * px;
}

__device__ bool solve_radius(const GridParams &p, Vec3f direction,
                             float rref, float pref, float &radius) {
    if (ray_potential(p, direction, rref) < pref) return false;
    float lo = 0.5f * rref, hi = rref, trial_potential = pref + 1.0f;
    for (int it = 0; it < 30 && trial_potential > pref; ++it) {
        lo = 0.5f * hi;
        trial_potential = ray_potential(p, direction, lo);
        if (trial_potential > pref) hi = lo;
    }
    if (trial_potential > pref) return false;

    // delta/10 can be below one FP32 ulp for small stars. This floor still
    // places the surface to substantially better than the requested phase
    // accuracy while preventing a stalled binary chop.
    const float acc = fmaxf(p.delta / 10.0f, 2.0f * FLT_EPSILON * rref);
    for (int it = 0; it < 100 && hi - lo > acc; ++it) {
        float mid = 0.5f * (lo + hi);
        if (ray_potential(p, direction, mid) < pref)
            lo = mid;
        else
            hi = mid;
    }
    radius = 0.5f * (lo + hi);
    return true;
}

__device__ __forceinline__ Vec3f earth_vector(float cosi, float sini,
                                               float phase) {
    float angle = TWOPI * phase;
    float s, c;
    sincosf(angle, &s, &c);
    return {sini * c, -sini * s, cosi};
}

__device__ bool line_sphere(Vec3f earth, Vec3f point, Vec3f centre,
                            float radius, float &lam1, float &lam2) {
    Vec3f d{point.x - centre.x, point.y - centre.y, point.z - centre.z};
    float b = dot(earth, d);
    if (b >= 0.0f) return false;
    float c = dot(d, d) - radius * radius;
    float disc = b * b - c;
    if (disc <= 0.0f) return false;
    float root = sqrtf(disc);
    lam2 = -b + root;
    lam1 = fmaxf(0.0f, c / lam2);
    return true;
}

__device__ bool orbital_sphere(float cosi, float sini, Vec3f point,
                               Vec3f centre, float radius, float &phi1,
                               float &phi2, float &lam1, float &lam2) {
    Vec3f d{point.x - centre.x, point.y - centre.y, point.z - centre.z};
    float pdist = sqrtf(d.x * d.x + d.y * d.y);
    float b = d.z * cosi - pdist * sini;
    if (b >= 0.0f) return false;
    float c = dot(d, d) - radius * radius;
    float disc = b * b - c;
    if (disc <= 0.0f) return false;
    float root = sqrtf(disc);
    lam2 = -b + root;
    lam1 = fmaxf(0.0f, c / lam2);
    if (c < 0.0f) {
        phi1 = 0.0f;
        phi2 = 1.0f;
    } else {
        float argument = (cosi * d.z + sqrtf(c)) / (sini * pdist);
        argument = fminf(1.0f, fmaxf(-1.0f, argument));
        float width = acosf(argument);
        float centre_phase = atan2f(d.y, -d.x);
        phi1 = (centre_phase - width) / TWOPI;
        phi1 -= floorf(phi1);
        phi2 = phi1 + 2.0f * width / TWOPI;
    }
    return true;
}

// The acos argument for an eclipse by a very small star differs from one by
// O(radius^2); FP32 loses several digits there. Evaluate just this inexpensive
// geometric filter in FP64 and cast the final phase interval back to FP32.
__device__ bool orbital_sphere_precise(float cosi_f, float sini_f,
                                       Vec3f point_f, Vec3f centre_f,
                                       float radius_f, float &phi1_f,
                                       float &phi2_f, float &lam1_f,
                                       float &lam2_f) {
    const double cosi = cosi_f, sini = sini_f, radius = radius_f;
    const double dx = static_cast<double>(point_f.x) - centre_f.x;
    const double dy = static_cast<double>(point_f.y) - centre_f.y;
    const double dz = static_cast<double>(point_f.z) - centre_f.z;
    const double pdist = sqrt(dx * dx + dy * dy);
    const double b = dz * cosi - pdist * sini;
    if (b >= 0.0) return false;
    const double c = dx * dx + dy * dy + dz * dz - radius * radius;
    const double disc = b * b - c;
    if (disc <= 0.0) return false;
    const double root = sqrt(disc);
    const double lam2 = -b + root;
    const double lam1 = fmax(0.0, c / lam2);
    double phi1, phi2;
    if (c < 0.0) {
        phi1 = 0.0; phi2 = 1.0;
    } else {
        double argument = (cosi * dz + sqrt(c)) / (sini * pdist);
        argument = fmin(1.0, fmax(-1.0, argument));
        const double width = acos(argument);
        const double centre_phase = atan2(dy, -dx);
        phi1 = (centre_phase - width) / static_cast<double>(TWOPI);
        phi1 -= floor(phi1);
        phi2 = phi1 + 2.0 * width / static_cast<double>(TWOPI);
    }
    phi1_f = static_cast<float>(phi1); phi2_f = static_cast<float>(phi2);
    lam1_f = static_cast<float>(lam1); lam2_f = static_cast<float>(lam2);
    return true;
}

struct Vec3d {
    double x, y, z;
};

__device__ __forceinline__ double dot(Vec3d a, Vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3d earth_vector64(const GridParams &p,
                                                double phase) {
    double s, c;
    sincos(6.283185307179586476925286766559 * phase, &s, &c);
    return {p.sini64 * c, -p.sini64 * s, p.cosi64};
}

__device__ __forceinline__ double secondary_potential64(
    const GridParams &p, Vec3d point) {
    const double mu = p.q64 / (1.0 + p.q64), comp = 1.0 - mu;
    const double x2y2 = point.x * point.x + point.y * point.y;
    const double r1sq = x2y2 + point.z * point.z;
    const double d1 = sqrt(r1sq);
    const double d2 = sqrt(r1sq + 1.0 - 2.0 * point.x);
    return -comp / d1 - mu / d2
           - p.spin2_64 * p.spin2_64 *
                 (0.5 + 0.5 * x2y2 - point.x)
           - comp * point.x;
}

__device__ bool line_sphere64(Vec3d earth, Vec3d point, double radius,
                              double &lam1, double &lam2) {
    Vec3d d{point.x - 1.0, point.y, point.z};
    const double b = dot(earth, d);
    if (b >= 0.0) return false;
    const double c = dot(d, d) - radius * radius;
    const double disc = b * b - c;
    if (disc <= 0.0) return false;
    const double root = sqrt(disc);
    lam2 = -b + root;
    lam1 = fmax(0.0, c / lam2);
    return true;
}

__device__ bool orbital_secondary64(const GridParams &p, Vec3d point,
                                    double &phi1, double &phi2,
                                    double &lam1, double &lam2) {
    const double dx = point.x - 1.0, dy = point.y, dz = point.z;
    const double pdist = sqrt(dx * dx + dy * dy);
    const double b = dz * p.cosi64 - pdist * p.sini64;
    if (b >= 0.0) return false;
    const double c = dx * dx + dy * dy + dz * dz
                     - p.rref2_64 * p.rref2_64;
    const double disc = b * b - c;
    if (disc <= 0.0) return false;
    const double root = sqrt(disc);
    lam2 = -b + root;
    lam1 = fmax(0.0, c / lam2);
    if (c < 0.0) {
        phi1 = 0.0; phi2 = 1.0;
    } else {
        double argument = (p.cosi64 * dz + sqrt(c)) /
                          (p.sini64 * pdist);
        argument = fmin(1.0, fmax(-1.0, argument));
        const double width = acos(argument);
        const double centre_phase = atan2(dy, -dx);
        phi1 = (centre_phase - width) /
               6.283185307179586476925286766559;
        phi1 -= floor(phi1);
        phi2 = phi1 + 2.0 * width /
               6.283185307179586476925286766559;
    }
    return true;
}

__device__ bool secondary_fblink64(const GridParams &p, Vec3d point,
                                   double phase, double acc) {
    const Vec3d earth = earth_vector64(p, phase);
    double lam1, lam2;
    if (!line_sphere64(earth, point, p.rref2_64, lam1, lam2)) return false;
    if (lam1 == 0.0) return true;
    auto line_potential = [&](double lambda) {
        Vec3d sample{point.x + lambda * earth.x,
                     point.y + lambda * earth.y,
                     point.z + lambda * earth.z};
        return secondary_potential64(p, sample);
    };
    constexpr double gr = 0.618033988749894848204586834366;
    double a = lam1, b = lam2;
    double x1 = b - gr * (b - a), x2 = a + gr * (b - a);
    double f1 = line_potential(x1), f2 = line_potential(x2);
    if (f1 <= p.pref2_64 || f2 <= p.pref2_64) return true;
    for (int it = 0; it < 60 && b - a > acc; ++it) {
        if (f1 < f2) {
            b = x2; x2 = x1; f2 = f1;
            x1 = b - gr * (b - a); f1 = line_potential(x1);
            if (f1 <= p.pref2_64) return true;
        } else {
            a = x1; x1 = x2; f1 = f2;
            x2 = a + gr * (b - a); f2 = line_potential(x2);
            if (f2 <= p.pref2_64) return true;
        }
    }
    return fmin(f1, f2) < p.pref2_64;
}

__device__ bool secondary_eclipse64(const GridParams &p, Vec3f point_f,
                                    float &ingress_f, float &egress_f) {
    const Vec3d point{point_f.x, point_f.y, point_f.z};
    double phi1, phi2, lam1, lam2;
    if (!orbital_secondary64(p, point, phi1, phi2, lam1, lam2)) return false;
    const double acc = 2.0 * sqrt(2.0 * 6.2831853071795864769 *
                                  (lam2 - lam1) * p.delta64);
    double inside = 0.5 * (phi1 + phi2);
    if (!secondary_fblink64(p, point, inside, acc)) {
        bool found = false;
        for (int sample = 1; sample <= 16; ++sample) {
            const double phase = phi1 + (phi2 - phi1) * sample / 17.0;
            if (secondary_fblink64(p, point, phase, acc)) {
                inside = phase; found = true; break;
            }
        }
        if (!found) return false;
    }
    double pin = inside, pout = phi1;
    for (int it = 0; it < 60 && fabs(pin - pout) > p.delta64; ++it) {
        double mid = 0.5 * (pin + pout);
        if (secondary_fblink64(p, point, mid, acc)) pin = mid;
        else pout = mid;
    }
    double ingress = 0.5 * (pin + pout);
    ingress -= floor(ingress);
    pin = inside; pout = phi2;
    for (int it = 0; it < 60 && fabs(pin - pout) > p.delta64; ++it) {
        double mid = 0.5 * (pin + pout);
        if (secondary_fblink64(p, point, mid, acc)) pin = mid;
        else pout = mid;
    }
    double egress = 0.5 * (pin + pout);
    egress -= floor(egress);
    if (egress < ingress) egress += 1.0;
    ingress_f = static_cast<float>(ingress);
    egress_f = static_cast<float>(egress);
    return true;
}

__device__ bool fast_fblink(const GridParams &p, int eclipsing_star,
                            float spin, Vec3f point, Vec3f earth,
                            float rref, float pref, float acc) {
    Vec3f centre = eclipsing_star == static_cast<int>(Roche::PRIMARY)
        ? Vec3f{0.0f, 0.0f, 0.0f} : Vec3f{1.0f, 0.0f, 0.0f};
    float lam1, lam2;
    if (!line_sphere(earth, point, centre, rref, lam1, lam2)) return false;
    if (lam1 == 0.0f) return true;

    auto line_potential = [&](float lambda) {
        return potential(p.q, eclipsing_star, spin,
                         add(point, scale(earth, lambda)));
    };

    // The Roche lobe is convex inside its reference sphere. A golden-section
    // minimisation is therefore sufficient here and maps much more cleanly to
    // one CUDA thread per face than the CPU's general-purpose dBrent path.
    constexpr float gr = 0.6180339887498948482f;
    float a = lam1, b = lam2;
    float x1 = b - gr * (b - a), x2 = a + gr * (b - a);
    float f1 = line_potential(x1), f2 = line_potential(x2);
    if (f1 <= pref || f2 <= pref) return true;
    for (int it = 0; it < 40 && b - a > acc; ++it) {
        if (f1 < f2) {
            b = x2; x2 = x1; f2 = f1;
            x1 = b - gr * (b - a); f1 = line_potential(x1);
            if (f1 <= pref) return true;
        } else {
            a = x1; x1 = x2; f1 = f2;
            x2 = a + gr * (b - a); f2 = line_potential(x2);
            if (f2 <= pref) return true;
        }
    }
    return fminf(f1, f2) < pref;
}

__device__ bool roche_eclipse(const GridParams &p, int eclipsing_star,
                              float spin, Vec3f point, float rref, float pref,
                              float &ingress, float &egress) {
    Vec3f centre = eclipsing_star == static_cast<int>(Roche::PRIMARY)
        ? Vec3f{0.0f, 0.0f, 0.0f} : Vec3f{1.0f, 0.0f, 0.0f};
    float phi1, phi2, lam1, lam2;
    if (!orbital_sphere(p.cosi, p.sini, point, centre, rref,
                        phi1, phi2, lam1, lam2))
        return false;
    float acc = 2.0f * sqrtf(2.0f * TWOPI * (lam2 - lam1) * p.delta);
    acc = fmaxf(acc, 8.0f * FLT_EPSILON);

    float inside = 0.5f * (phi1 + phi2);
    if (!fast_fblink(p, eclipsing_star, spin, point,
                     earth_vector(p.cosi, p.sini, inside), rref, pref, acc)) {
        bool found = false;
        for (int sample = 1; sample <= 16; ++sample) {
            float phase = phi1 + (phi2 - phi1) * sample / 17.0f;
            if (fast_fblink(p, eclipsing_star, spin, point,
                            earth_vector(p.cosi, p.sini, phase),
                            rref, pref, acc)) {
                inside = phase;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    float pin = inside, pout = phi1;
    for (int it = 0; it < 40 && fabsf(pin - pout) > p.delta; ++it) {
        float mid = 0.5f * (pin + pout);
        if (fast_fblink(p, eclipsing_star, spin, point,
                        earth_vector(p.cosi, p.sini, mid),
                        rref, pref, acc))
            pin = mid;
        else
            pout = mid;
    }
    ingress = 0.5f * (pin + pout);
    ingress -= floorf(ingress);

    pin = inside; pout = phi2;
    for (int it = 0; it < 40 && fabsf(pin - pout) > p.delta; ++it) {
        float mid = 0.5f * (pin + pout);
        if (fast_fblink(p, eclipsing_star, spin, point,
                        earth_vector(p.cosi, p.sini, mid),
                        rref, pref, acc))
            pin = mid;
        else
            pout = mid;
    }
    egress = 0.5f * (pin + pout);
    egress -= floorf(egress);
    if (egress < ingress) egress += 1.0f;
    return true;
}

__global__ void build_faces_kernel(const float *directions, int n,
                                   GridParams params, float *output,
                                   unsigned char *status) {
    int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3f direction{directions[i], directions[n + i], directions[2 * n + i]};
    const bool primary = params.which_star == static_cast<int>(Roche::PRIMARY);
    const bool roche_self = primary ? params.roche1 : params.roche2;
    const float self_radius = primary ? params.r1 : params.r2;
    const float self_rref = primary ? params.rref1 : params.rref2;
    const float self_pref = primary ? params.pref1 : params.pref2;
    const float self_spin = primary ? params.spin1 : params.spin2;

    float radius = self_radius;
    if (roche_self && !solve_radius(params, direction, self_rref,
                                    self_pref, radius)) {
        status[i] = 2;
        return;
    }

    Vec3f centre = primary ? Vec3f{0.0f, 0.0f, 0.0f}
                           : Vec3f{1.0f, 0.0f, 0.0f};
    Vec3f point = add(centre, scale(direction, radius));
    Vec3f normal = direction;
    float gravity = 1.0f;
    if (roche_self) {
        normal = potential_gradient(params.q, params.which_star,
                                    self_spin, point);
        gravity = sqrtf(dot(normal, normal));
        normal = scale(normal, 1.0f / gravity);
    }

    float ingress = 0.0f, egress = 0.0f;
    bool is_eclipsed = false;
    if (params.eclipse) {
        const int other = primary ? static_cast<int>(Roche::SECONDARY)
                                  : static_cast<int>(Roche::PRIMARY);
        const bool roche_other = primary ? params.roche2 : params.roche1;
        const float other_radius = primary ? params.r2 : params.r1;
        const float other_rref = primary ? params.rref2 : params.rref1;
        const float other_pref = primary ? params.pref2 : params.pref1;
        const float other_spin = primary ? params.spin2 : params.spin1;
        Vec3f other_centre = primary ? Vec3f{1.0f, 0.0f, 0.0f}
                                     : Vec3f{0.0f, 0.0f, 0.0f};
        // At radii below 2% of the separation, Roche distortion is O(r^3)
        // and below FP32 resolution. Using the physical radius as a sphere
        // is both more accurate and much better conditioned than evaluating
        // the singular Roche potential of a tiny companion in FP32.
        if (roche_other && other_radius >= 0.02f) {
            is_eclipsed = roche_eclipse(params, other, other_spin, point,
                                         other_rref, other_pref,
                                         ingress, egress);
        } else if (roche_other &&
                   other == static_cast<int>(Roche::SECONDARY)) {
            is_eclipsed = secondary_eclipse64(params, point,
                                               ingress, egress);
        } else if (other_radius < 0.02f) {
            float lam1, lam2;
            is_eclipsed = orbital_sphere_precise(
                params.cosi, params.sini, point, other_centre, other_radius,
                ingress, egress, lam1, lam2);
        } else {
            float lam1, lam2;
            is_eclipsed = orbital_sphere(params.cosi, params.sini, point,
                                          other_centre, other_radius,
                                          ingress, egress, lam1, lam2);
        }
    }

    output[i] = radius;
    output[n + i] = normal.x;
    output[2 * n + i] = normal.y;
    output[3 * n + i] = normal.z;
    output[4 * n + i] = gravity / params.gref;
    output[5 * n + i] = ingress;
    output[6 * n + i] = egress;
    status[i] = is_eclipsed ? 1 : 0;
}

class GridCudaContext {
public:
    ~GridCudaContext() {
        if (profile_ && calls_)
            std::fprintf(stderr,
                         "LCURVE CUDA grid profile: %zu calls, pack %.3f s, "
                         "H2D+kernel+D2H %.3f s, unpack %.3f s\n",
                         calls_, pack_seconds_, device_seconds_, unpack_seconds_);
    }

    bool initialise() {
        if (initialised_) return enabled_;
        initialised_ = true;
        verbose_ = std::getenv("LCURVE_CUDA_VERBOSE") != nullptr;
        profile_ = std::getenv("LCURVE_CUDA_GRID_PROFILE") != nullptr;
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) {
            cudaGetLastError();
            return false;
        }
        int device = 0;
        if (const char *env = std::getenv("LCURVE_CUDA_DEVICE"))
            device = std::max(0, std::atoi(env));
        if (device >= count) device = 0;
        enabled_ = cudaSetDevice(device) == cudaSuccess;
        return enabled_;
    }

    bool reserve(std::size_t n) {
        host_input.resize(3 * n);
        host_output.resize(7 * n);
        host_status.resize(n);
        return device_input.reserve(3 * n) && device_output.reserve(7 * n) &&
               device_status.reserve(n);
    }

    void record(double pack, double device, double unpack) {
        ++calls_;
        if (!profile_) return;
        pack_seconds_ += pack;
        device_seconds_ += device;
        unpack_seconds_ += unpack;
    }

    bool verbose() const { return verbose_; }
    std::uint64_t calls() const { return calls_; }
    DeviceBuffer<float> device_input, device_output;
    DeviceBuffer<unsigned char> device_status;
    std::vector<float> host_input, host_output;
    std::vector<unsigned char> host_status;

private:
    bool initialised_ = false, enabled_ = false;
    bool verbose_ = false, profile_ = false;
    std::size_t calls_ = 0;
    double pack_seconds_ = 0.0, device_seconds_ = 0.0, unpack_seconds_ = 0.0;
};

GridCudaContext &grid_context() {
    static thread_local GridCudaContext context;
    return context;
}

bool env_true(const char *value) {
    if (!value) return false;
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text == "1" || text == "on" || text == "true" || text == "yes";
}

} // unnamed namespace

bool Lcurve::cuda_star_grid_enabled() {
    const char *cuda = std::getenv("LCURVE_CUDA");
    if (cuda) {
        std::string mode(cuda);
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (mode == "0" || mode == "off" || mode == "false" ||
            mode == "cpu")
            return false;
    }
    const char *grid = std::getenv("LCURVE_CUDA_GRID");
    const bool requested = grid ? env_true(grid) : env_true(cuda);
    return requested && grid_context().initialise();
}

bool Lcurve::cuda_build_star_faces(
    std::vector<Point> &star, int first_face,
    const std::vector<double> &dx, const std::vector<double> &dy,
    const std::vector<double> &dz, const std::vector<double> &area_scale,
    Roche::STAR which_star, double q, double iangle, double r1, double r2,
    double rref1, double rref2, bool roche1, bool roche2,
    double spin1, double spin2, bool eclipse, double gref,
    double pref1, double pref2, double delta) {
    const std::size_t n = dx.size();
    if (!n || dy.size() != n || dz.size() != n || area_scale.size() != n)
        return false;
    GridCudaContext &context = grid_context();
    if (!context.initialise() || !context.reserve(n)) return false;

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        context.host_input[i] = static_cast<float>(dx[i]);
        context.host_input[n + i] = static_cast<float>(dy[i]);
        context.host_input[2 * n + i] = static_cast<float>(dz[i]);
    }
    auto t1 = Clock::now();

    GridParams params{};
    params.q = static_cast<float>(q);
    float ri = static_cast<float>(iangle * PI / 180.0);
    params.cosi = cosf(ri); params.sini = sinf(ri);
    params.r1 = static_cast<float>(r1); params.r2 = static_cast<float>(r2);
    params.rref1 = static_cast<float>(rref1);
    params.rref2 = static_cast<float>(rref2);
    params.spin1 = static_cast<float>(spin1);
    params.spin2 = static_cast<float>(spin2);
    params.gref = static_cast<float>(gref);
    params.pref1 = static_cast<float>(pref1);
    params.pref2 = static_cast<float>(pref2);
    params.delta = static_cast<float>(delta);
    params.which_star = static_cast<int>(which_star);
    params.roche1 = roche1; params.roche2 = roche2; params.eclipse = eclipse;
    params.q64 = q;
    params.cosi64 = std::cos(iangle * 3.14159265358979323846 / 180.0);
    params.sini64 = std::sin(iangle * 3.14159265358979323846 / 180.0);
    params.rref2_64 = rref2;
    params.spin2_64 = spin2;
    params.pref2_64 = pref2;
    params.delta64 = delta;

    cudaError_t err = cudaMemcpy(context.device_input.get(),
                                 context.host_input.data(),
                                 3 * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;
    int blocks = static_cast<int>((n + THREADS - 1) / THREADS);
    build_faces_kernel<<<blocks, THREADS>>>(context.device_input.get(),
                                            static_cast<int>(n), params,
                                            context.device_output.get(),
                                            context.device_status.get());
    err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaMemcpy(context.host_output.data(), context.device_output.get(),
                         7 * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
        err = cudaMemcpy(context.host_status.data(), context.device_status.get(),
                         n * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        if (context.verbose())
            std::cerr << "LCURVE CUDA grid kernel failed: "
                      << cudaGetErrorString(err) << "; using CPU grid path\n";
        cudaGetLastError();
        return false;
    }
    auto t2 = Clock::now();

    if (std::find(context.host_status.begin(), context.host_status.end(), 2) !=
        context.host_status.end())
        return false;

    const bool validate_eclipse =
        env_true(std::getenv("LCURVE_CUDA_GRID_VALIDATE"));
    const bool cpu_eclipse = validate_eclipse ||
        env_true(std::getenv("LCURVE_CUDA_GRID_CPU_ECLIPSE"));
    const double ri_double = Subs::deg2rad(iangle);
    const double cosi_double = std::cos(ri_double), sini_double = std::sin(ri_double);
    std::size_t eclipse_mismatches = 0, eclipse_matches = 0;
    double max_ingress_error = 0.0, max_egress_error = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double radius = context.host_output[i];
        Subs::Vec3 direction(dx[i], dy[i], dz[i]);
        Subs::Vec3 centre = which_star == Roche::PRIMARY
            ? Subs::Vec3(0., 0., 0.) : Subs::Vec3(1., 0., 0.);
        Subs::Vec3 position = centre + radius * direction;
        Subs::Vec3 normal(context.host_output[n + i],
                          context.host_output[2 * n + i],
                          context.host_output[3 * n + i]);
        double area = area_scale[i] * radius * radius /
                      Subs::dot(direction, normal);
        Point::etype eclipses;
        if (cpu_eclipse && eclipse) {
            double ingress, egress, lam1, lam2;
            bool hidden;
            if (which_star == Roche::PRIMARY) {
                hidden = roche2
                    ? Roche::ingress_egress(q, Roche::SECONDARY, spin2,
                                             r2 / (1. - Roche::xl12(q, spin2)),
                                             iangle, delta, position,
                                             ingress, egress, rref2, pref2)
                    : Roche::sphere_eclipse(cosi_double, sini_double, position,
                                             Subs::Vec3(1., 0., 0.), r2,
                                             ingress, egress, lam1, lam2);
            } else {
                hidden = roche1
                    ? Roche::ingress_egress(q, Roche::PRIMARY, spin1,
                                             r1 / Roche::xl11(q, spin1),
                                             iangle, delta, position,
                                             ingress, egress, rref1, pref1)
                    : Roche::sphere_eclipse(cosi_double, sini_double, position,
                                             Subs::Vec3(0., 0., 0.), r1,
                                             ingress, egress, lam1, lam2);
            }
            if (validate_eclipse) {
                const bool gpu_hidden = context.host_status[i] == 1;
                if (hidden != gpu_hidden) {
                    ++eclipse_mismatches;
                } else if (hidden) {
                    ++eclipse_matches;
                    max_ingress_error = std::max(
                        max_ingress_error,
                        std::abs(ingress - context.host_output[5 * n + i]));
                    max_egress_error = std::max(
                        max_egress_error,
                        std::abs(egress - context.host_output[6 * n + i]));
                }
            }
            if (hidden) eclipses.emplace_back(ingress, egress);
        } else if (context.host_status[i] == 1) {
            eclipses.emplace_back(context.host_output[5 * n + i],
                                  context.host_output[6 * n + i]);
        }
        star[first_face + i] = Point(position, normal, area,
                                     context.host_output[4 * n + i], eclipses);
    }
    if (validate_eclipse)
        std::cerr << "LCURVE CUDA eclipse validation for " << n
                  << " faces: mismatched visibility=" << eclipse_mismatches
                  << ", matched eclipsed=" << eclipse_matches
                  << ", max ingress error=" << max_ingress_error
                  << ", max egress error=" << max_egress_error << '\n';
    auto t3 = Clock::now();
    auto seconds = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    context.record(seconds(t0, t1), seconds(t1, t2), seconds(t2, t3));
    return true;
}

std::uint64_t Lcurve::cuda_grid_evaluation_count() {
    return grid_context().calls();
}
