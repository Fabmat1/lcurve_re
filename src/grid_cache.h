#pragma once
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <list>

class GridCache {
public:
    struct CachedModel {
        std::vector<double> model_values;  // The actual light curve
        double chisq;
        std::chrono::steady_clock::time_point last_access;
    };
    
    GridCache(int ndim, const std::vector<double>& initial_params,
              const std::vector<double>& step_sizes, 
              const std::vector<std::pair<double,double>>& param_limits,
              size_t max_size = 50000)
        : ndim_(ndim), initial_params_(initial_params), 
          step_sizes_(step_sizes), param_limits_(param_limits),
          max_cache_size_(max_size), grid_spacing_(step_sizes.size()) {
        
        // Set grid spacing to 10x the step size
        for (size_t i = 0; i < step_sizes.size(); ++i) {
            grid_spacing_[i] = step_sizes[i];
        }
    }
    
    // Convert continuous parameter to grid index (centered on initial params)
    std::vector<int> param_to_grid_index(const std::vector<double>& params) const {
        std::vector<int> indices(ndim_);
        for (int i = 0; i < ndim_; ++i) {
            indices[i] = std::round((params[i] - initial_params_[i]) / grid_spacing_[i]);
        }
        return indices;
    }
    
    // Convert grid index to parameter value (centered on initial params)
    std::vector<double> grid_index_to_param(const std::vector<int>& indices) const {
        std::vector<double> params(ndim_);
        for (int i = 0; i < ndim_; ++i) {
            params[i] = initial_params_[i] + indices[i] * grid_spacing_[i];
            // Ensure we stay within limits
            params[i] = std::max(param_limits_[i].first, 
                               std::min(params[i], param_limits_[i].second));
        }
        return params;
    }
    
    // Hash function for grid indices
    std::string make_key(const std::vector<int>& indices) const {
        std::string key;
        for (int idx : indices) {
            key += std::to_string(idx) + "_";
        }
        return key;
    }
    
    // Check if a grid node exists in cache
    bool has_node(const std::vector<int>& grid_idx) const {
        return cache_.find(make_key(grid_idx)) != cache_.end();
    }
    
    // Get cached model (returns true if found)
    bool get_cached_model(const std::vector<int>& grid_idx, 
                         CachedModel& model) {
        std::string key = make_key(grid_idx);
        auto it = cache_.find(key);
        
        if (it != cache_.end()) {
            // Update access time and move to front of LRU list
            it->second.last_access = std::chrono::steady_clock::now();
            
            // Move to front in access list
            auto list_it = std::find(access_order_.begin(), access_order_.end(), key);
            if (list_it != access_order_.end()) {
                access_order_.erase(list_it);
            }
            access_order_.push_front(key);
            
            model = it->second;
            hits_++;
            return true;
        }
        misses_++;
        return false;
    }
    
    // Add a computed model to cache
    void add_model(const std::vector<int>& grid_idx, 
                  const CachedModel& model) {
        std::string key = make_key(grid_idx);
        
        // Check if we need to evict
        if (cache_.size() >= max_cache_size_ && cache_.find(key) == cache_.end()) {
            // Remove least recently used
            if (!access_order_.empty()) {
                std::string lru_key = access_order_.back();
                access_order_.pop_back();
                cache_.erase(lru_key);
            }
        }
        
        // Add or update
        cache_[key] = model;
        cache_[key].last_access = std::chrono::steady_clock::now();
        
        // Update access order
        auto list_it = std::find(access_order_.begin(), access_order_.end(), key);
        if (list_it != access_order_.end()) {
            access_order_.erase(list_it);
        }
        access_order_.push_front(key);
    }
    
    // Find nearest cached neighbors for interpolation
    bool find_interpolation_neighbors(const std::vector<double>& params,
                                     std::vector<std::pair<std::vector<int>, double>>& neighbors,
                                     int max_neighbors = 8) const {
        neighbors.clear();
        auto target_idx = param_to_grid_index(params);
        
        // Search in expanding hypercube around target
        for (int radius = 0; radius <= 2 && neighbors.size() < max_neighbors; ++radius) {
            search_radius(target_idx, radius, neighbors, max_neighbors);
        }
        
        return !neighbors.empty();
    }
    
    // Get cache statistics
    void print_stats() const {
        double hit_rate = (hits_ + misses_ > 0) ? 
                         100.0 * hits_ / (hits_ + misses_) : 0.0;
        std::cout << "Cache: " << cache_.size() << "/" << max_cache_size_ 
                  << " entries, Hit rate: " << hit_rate << "%" << std::endl;
    }

    // Get grid spacing for a dimension
    double get_grid_spacing(int dim) const {
        return (dim >= 0 && dim < ndim_) ? grid_spacing_[dim] : 0.0;
    }
    
    // Calculate normalized distance from a point to nearest grid point
    double distance_to_nearest_grid_point(const std::vector<double>& params) const {
        auto grid_idx = param_to_grid_index(params);
        auto grid_params = grid_index_to_param(grid_idx);
        
        double max_dist = 0.0;
        for (int i = 0; i < ndim_; ++i) {
            double dist = std::abs(params[i] - grid_params[i]) / grid_spacing_[i];
            max_dist = std::max(max_dist, dist);
        }
        return max_dist;
    }
    
    // Check if a point is close enough to use cached grid value
    bool is_close_to_grid_point(const std::vector<double>& params, 
                                double threshold = 0.5) const {
        return distance_to_nearest_grid_point(params) < threshold;
    }
    
private:
    void search_radius(const std::vector<int>& center, int radius,
                      std::vector<std::pair<std::vector<int>, double>>& neighbors,
                      int max_neighbors) const {
        if (radius == 0) {
            // Check exact point
            if (has_node(center)) {
                neighbors.push_back({center, 0.0});
            }
            return;
        }
        
        // Generate all points at Manhattan distance = radius
        std::vector<int> current = center;
        search_radius_recursive(center, current, 0, radius, 0, neighbors, max_neighbors);
    }
    
    void search_radius_recursive(const std::vector<int>& center,
                                std::vector<int>& current,
                                int dim, int remaining_dist, double euclidean_dist,
                                std::vector<std::pair<std::vector<int>, double>>& neighbors,
                                int max_neighbors) const {
        if (neighbors.size() >= max_neighbors) return;
        
        if (dim == ndim_) {
            if (remaining_dist == 0 && has_node(current)) {
                neighbors.push_back({current, std::sqrt(euclidean_dist)});
            }
            return;
        }
        
        // Try different displacements for this dimension
        for (int d = -remaining_dist; d <= remaining_dist; ++d) {
            if (std::abs(d) > remaining_dist) continue;
            
            current[dim] = center[dim] + d;
            double new_dist = euclidean_dist + d * d;
            search_radius_recursive(center, current, dim + 1, 
                                  remaining_dist - std::abs(d), new_dist,
                                  neighbors, max_neighbors);
        }
        current[dim] = center[dim];  // Reset
    }
    
    int ndim_;
    std::vector<double> initial_params_;
    std::vector<double> step_sizes_;
    std::vector<double> grid_spacing_;
    std::vector<std::pair<double,double>> param_limits_;
    size_t max_cache_size_;
    
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
    
    std::unordered_map<std::string, CachedModel> cache_;
    std::list<std::string> access_order_;  // For LRU
};