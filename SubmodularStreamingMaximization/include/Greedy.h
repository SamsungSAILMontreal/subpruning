#ifndef GREEDY_H
#define GREEDY_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

/**
 * @brief  The Greedy optimizer for submodular functions. It rates the marginal gain of each element and picks that element with the largest gain. This process is repeated until it K elements have been selected:
 *  - Stream:  No
 *  - Solution: 1 - exp(1)
 *  - Runtime: O(N * K)
 *  - Memory: O(K)
 *  - Function Queries per Element: O(1)
 *  - Function Types: nonnegative submodular functions
 * 
 * See also :
 *   - Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions-I. Mathematical Programming, 14(1), 265–294. https://doi.org/10.1007/BF01588971
 * @note   
 */

template <class T>
class Greedy : public SubmodularOptimizer<T> {
public:
    int s;
    /**
     * @brief Construct a new Greedy object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that the `clone' function is used to construct a new SubmodularFunction which is owned by this object. If you implement a custom SubmodularFunction make sure that everything you need is actually cloned / copied.  
     * @param s randomly sample s remaining elements to evaluate at each iteration of greedy
     */
    Greedy(unsigned int K, SubmodularFunction<T> & f, unsigned int s=std::numeric_limits<int>::max()) : SubmodularOptimizer<T>(K,f), s(s) {
        assert(s != 0 && "s should be greater or equal to 1");
    }


    /**
     * @brief Construct a new Greedy object
     * 
     * @param K The cardinality constraint you of the optimization problem, that is the number of items selected.
     * @param f The function which should be maximized. Note, that this parameter is likely moved and not copied. Thus, if you construct multiple optimizers with the __same__ function they all reference the __same__ function. This can be very efficient for state-less functions, but may lead to weird side effects if f keeps track of a state.
     */
    Greedy(unsigned int K, std::function<data_t (std::vector<T> const &)> f, unsigned int s=std::numeric_limits<int>::max()) : SubmodularOptimizer<T>(K,f), s(s) {
        assert(s != 0 && "s should be greater or equal to 1");
    }

    /**
     * @brief Pick that element with the largest marginal gain in the entire dataset. Repeat this until K element have been selected. You can access the solution via `get_solution`
     * 
     * @param X A constant reference to the entire data set
     * @param iterations: Has no effect. Greedy iterates K times over the entire dataset in any case.
     */
    void fit(std::vector<T> const & X, std::vector<idx_t> const & ids, unsigned int iterations = 1) {
    //void fit(std::vector<std::vector<data_t>> const & X, unsigned int iterations = 1) {
        //std::cout<<"sample produced using global generator = "<<global_generator()<<std::endl;
        std::vector<unsigned int> remaining(X.size());
        std::iota(remaining.begin(), remaining.end(), 0);
        data_t fcur = 0;
        progressbar bar(this->K);
        while(this->solution.size() < this->K && remaining.size() > 0) {
            bar.update();
            int subset_size = std::min(this->s, int(remaining.size()));
            //std::cout<<"subset size = "<<subset_size<<std::endl;
            std::vector<data_t> fvals;
            fvals.reserve(subset_size);
            std::vector<unsigned int> remaining_subset;
            std::sample(remaining.begin(), remaining.end(), std::back_inserter(remaining_subset), subset_size, global_generator);
            //std::cout<<"remaining subset = "<<remaining_subset<<std::endl;
            // Technically the Greedy algorithms picks that element with largest gain. This is equivalent to picking that
            // element which results in the largest function value. There is no need to explicitly compute the gain
            for (auto i : remaining_subset) {
                data_t ftmp = this->f->peek(this->solution, X[i], this->solution.size());
                fvals.push_back(ftmp);
            }

            unsigned int max_element = std::distance(fvals.begin(),std::max_element(fvals.begin(), fvals.end()));
            fcur = fvals[max_element];
            unsigned int max_idx = remaining[max_element];
            
            // Copy new vector into solution vector
            this->f->update(this->solution, X[max_idx], this->solution.size());
            //solution.push_back(std::vector<data_t>(X[max_idx]));
            this->solution.push_back(X[max_idx]);
            if (ids.size() >= max_idx) {
                this->ids.push_back(max_idx);
            }
            remaining.erase(remaining.begin()+max_element);
        }

        this->fval = fcur;
        this->is_fitted = true;
    }

    void fit(std::vector<T> const & X, unsigned int iterations = 1) {
        std::vector<idx_t> ids;
        fit(X,ids,iterations);
    }


    /**
     * @brief Throws an exception when called. Greedy does not support streaming!
     * 
     * @param x A constant reference to the next object on the stream.
     */
    void next(T const &x, std::optional<idx_t> id = std::nullopt) {
        throw std::runtime_error("Greedy does not support streaming data, please use fit().");
    }

    
};

#endif // GREEDY_H
