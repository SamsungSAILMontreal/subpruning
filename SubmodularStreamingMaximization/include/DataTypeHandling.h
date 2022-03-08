#ifndef DATATYPEHANDLING_H_
#define DATATYPEHANDLING_H_

#include <vector>

//#include "Eigen/Eigen"
//#include "float.h"
//#include <memory>

typedef double data_t;
typedef long idx_t;

// struct Data {
//     std::vector<data_t> x;
//     unsigned long id = 0;
// };

/*typedef Eigen::MatrixXd MatrixX;
typedef Eigen::VectorXd VectorX;

typedef Eigen::Ref<VectorX, 0, Eigen::InnerStride<>> VectorXRef;
typedef Eigen::Ref<const VectorX, 0, Eigen::InnerStride<>> ConstVectorXRef;
typedef std::shared_ptr<MatrixX> SharedMatrixX;*/

// print pairs
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2> p)
{
    os << "(" << p.first << "," << p.second << ") ";
    return os;
}

// print vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> & vec)
{
    for(auto & elt: vec){
        os << elt << " ";
    }
    os << std::endl;
    return os;
}

#endif
