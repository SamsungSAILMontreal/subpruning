#ifndef INDEPENDENT_SET_IMPROVEMENT_H
#define INDEPENDENT_SET_IMPROVEMENT_H

#include "DataTypeHandling.h"
#include "SubmodularOptimizer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <string>
#include <queue>

#include <iostream>

template <class T>
class IndependentSetImprovement : public SubmodularOptimizer<T> {

protected:
    struct Pair {
        data_t weight;
        unsigned int idx;

        Pair(data_t _weight, unsigned int _idx) {
            weight = _weight;
            idx = _idx;
        }

        // bool operator > (const Pair &other) const { 
        //     return weight < other.weight; 
        // } 

        bool operator < (const Pair &other) const { 
            return weight > other.weight; 
        } 
    };

    std::priority_queue<Pair> weights; 
public:

    IndependentSetImprovement(unsigned int K, SubmodularFunction<T> & f) : SubmodularOptimizer<T>(K,f)  {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }

    IndependentSetImprovement(unsigned int K, std::function<data_t (std::vector<T> const &)> f) : SubmodularOptimizer<T>(K,f) {
        // assert(("T should at-least be 1 or greater.", T >= 1));
    }
    
    void next(T const &x, std::optional<idx_t> const id = std::nullopt) {
        unsigned int Kcur = this->solution.size();
        
        if (Kcur < this->K) {
            data_t w = this->f->peek(this->solution, x, this->solution.size()) - this->fval;
            this->f->update(this->solution, x, this->solution.size());
            this->solution.push_back(x);
            this->ids.push_back(id.value());
            weights.push(Pair(w, Kcur));
        } else {
            Pair to_replace = weights.top();
            data_t w = this->f->peek(this->solution, x, this->solution.size()) - this->fval;
            if (w > 2*to_replace.weight) {
                this->f->update(this->solution, x, to_replace.idx);
                this->solution[to_replace.idx] = x;
                if (id.has_value()) this->ids[to_replace.idx] = id.value();
                weights.pop();
                weights.push(Pair(w, to_replace.idx));
            }
        }
        this->fval = this->f->operator()(this->solution);
        this->is_fitted = true;
    }
};

#endif