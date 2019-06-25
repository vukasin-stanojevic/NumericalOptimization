#ifndef PROJEKATC___LINEAR_ALGEBRA_H
#define PROJEKATC___LINEAR_ALGEBRA_H


#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <cmath>

namespace la {

template<class T>
class mat;

template<class T>
class vec {
protected:
    size_t n;
    T* a;

    void check_dims(const vec& b) const {
        if (n != b.n)
            throw "operand size mismatch";
    }

public:

    // Generic OOP stvari

    vec() : n(0), a(nullptr) {}

    vec(size_t n) : n(n), a(new T[n]) {}

    vec(size_t n, const T& val) : n(n), a(new T[n]) {
        for (size_t i=0; i<n; i++)
            a[i] = val;
    }

    ~vec() {
        delete[] a;
    }

    vec(const vec& b) : n(b.n), a(new T[n]) {
        for (size_t i=0; i<n; i++)
            a[i] = b.a[i];
    }

    vec(vec&& b) : n(b.n), a(b.a) {
        b.a = nullptr;
        b.n = 0;
    }

    template<class U>
    vec(std::initializer_list<U> b) : n(b.size()), a(new T[n]) {
        auto it = b.begin();
        size_t i = 0;
        while (it != b.end()) {
            a[i++] = *it;
            ++it;
        }
    }

    vec& operator= (const vec& b) {
        if (&b == this)
            return *this;

        delete[] a;
        n = b.n;
        a = new T[n];
        for (size_t i=0; i<n; i++)
            a[i] = b.a[i];
        return *this;
    }

    vec& operator= (vec&& b) {
        delete[] a;
        n = b.n;
        a = new T[n];
        for (size_t i=0; i<n; i++)
            a[i] = b.a[i];
        b.a = nullptr;
        b.a = 0;
        return *this;
    }

    // Generic C++ STL-like stvari

    size_t size() const { return n; }

    bool empty() const { return n == 0; }

    const T* begin() const { return a; }

    const T* end() const { return a+n; }

    T* begin() { return a; }

    T* end() { return a+n; }

    T& operator[] (size_t i) { return a[i]; }
    const T& operator[] (size_t i) const { return a[i]; }

    // Skalarni compound operatori

    vec& operator+= (const T& x) {
        for (size_t i=0; i<n; i++)
            a[i] += x;
        return *this;
    }

    vec& operator-= (const T& x) {
        for (size_t i=0; i<n; i++)
            a[i] -= x;
        return *this;
    }

    vec& operator*= (const T& x) {
        for (size_t i=0; i<n; i++)
            a[i] *= x;
        return *this;
    }

    vec& operator/= (const T& x) {
        for (size_t i=0; i<n; i++)
            a[i] /= x;
        return *this;
    }

    // Skalarni obicni operatori

    vec operator+ (const T& x) const {
        vec tmp = *this;
        tmp += x;
        return tmp;
    }

    vec operator- (const T& x) const {
        vec tmp = *this;
        tmp -= x;
        return tmp;
    }

    vec operator* (const T& x) const {
        vec tmp = *this;
        tmp *= x;
        return tmp;
    }

    vec operator/ (const T& x) const {
        vec tmp = *this;
        tmp /= x;
        return tmp;
    }

    // Vektorski pointwise compound operatori
    vec& operator+= (const vec& x) {
        check_dims(x);
        for (size_t i=0; i<n; i++)
            a[i] += x.a[i];
        return *this;
    }

    vec& operator-= (const vec& x) {
        check_dims(x);
        for (size_t i=0; i<n; i++)
            a[i] -= x.a[i];
        return *this;
    }

    vec& operator*= (const vec& x) {
        check_dims(x);
        for (size_t i=0; i<n; i++)
            a[i] *= x.a[i];
        return *this;
    }

    vec& operator/= (const vec& x) {
        check_dims(x);
        for (size_t i=0; i<n; i++)
            a[i] /= x.a[i];
        return *this;
    }

    // Vektorski obicni pointwise operatori
    vec operator+ (const vec& x) const {
        vec tmp = *this;
        tmp += x;
        return tmp;
    }

    vec operator- (const vec& x) const {
        vec tmp = *this;
        tmp -= x;
        return tmp;
    }

    vec operator* (const vec& x) const {
        vec tmp = *this;
        tmp *= x;
        return tmp;
    }

    vec operator/ (const vec& x) const {
        vec tmp = *this;
        tmp /= x;
        return tmp;
    }

    vec operator- () const {
        return *this * -1;
    }

    // Skalarni proizvod
    T inner(const vec& x) const {
        check_dims(x);
        T z = 0;
        for (size_t i=0; i<n; i++)
            z += a[i] * x.a[i];
        return z;
    }

    // Alias za inner
    T dot(const vec& x) const {
        return inner(x);
    }

    // Outer product vektora, rezultat je matrica
    mat<T> outer(const vec& x) const {
        if (n == 0 || x.n == 0)
            return mat<T>();

        mat<T> z(n, x.n);

        for (size_t i=0; i<n; i++)
            for (size_t j=0; j<x.n; j++)
                z[i][j] = a[i] * x.a[j];

        return z;
    }
};

template<class T>
std::ostream& operator<< (std::ostream& os, const vec<T>& v) {
    os << "[";
    for (size_t i=0; i<v.size(); i++) {
        os << v[i];
        if (i+1 != v.size())
            os << ", ";
    }
    os << "]";
    return os;
}

template<class U>
class mat {
protected:
    vec<vec<U>> a;

    void check_dims(const mat& b) const {
        if (rows() != b.rows() || cols() != b.cols())
            throw "operand size mismatch";
    }

    class vec_proxy {
    private:
        vec<U>& v;
    public:
        friend class mat;

        vec_proxy(vec<U>& v) : v(v) {}

        U& operator[] (size_t i) {
            return v[i];
        }
    };

    class const_vec_proxy {
    private:
        const vec<U>& v;
    public:
        friend class mat;

        const_vec_proxy(const vec<U>& v) : v(v) {}

        const U& operator[] (size_t i) const {
            return v[i];
        }
    };
public:
    template<class T>
    friend mat<T> operator*(const T val,const mat<T>& matrix){
        return matrix*val;
    }
    template<class T>
    friend mat<T> operator+(const T val,const mat<T>& matrix){
        return matrix+val;
    }
    template<class T>
    friend mat<T> operator-(const T val,const mat<T>& matrix){
        return matrix-val;
    }

public:
    mat() : a() {}

    mat(size_t n, size_t m) : a(n, vec<U>(m)) {}

    mat(size_t n, size_t m, const U& val) :
            a(n, vec<U>(m, val)) {}

    size_t rows() const { return a.size(); }

    size_t cols() const {
        if (rows() == 0)
            return 0;
        return a[0].size();
    }

    mat(std::initializer_list<vec<U>> b) {
        if (b.size() == 0)
            return;

        auto it0 = b.begin();
        auto it1 = it0;
        ++it1;

        while (it1 != b.end()) {
            if (it0->size() != it1->size())
                throw "row size mismatch";
            ++it0;
            ++it1;
        }

        if (b.begin()->size() == 0)
            return;

        it0 = b.begin();
        size_t i = 0;

        a = vec<vec<U>>(b.size());
        while (it0 != b.end()) {
            a[i++] = *it0;
            ++it0;
        }

    }

    size_t size() const { return rows() * cols(); }

    bool empty() const { return size() == 0; }

    vec_proxy operator[] (size_t i) { return a[i]; }
    const_vec_proxy operator[] (size_t i) const { return a[i]; }

    // Skalarni compound operatori

    mat& operator+= (const U& x) {
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] += x;
        return *this;
    }

    mat& operator-= (const U& x) {
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] -= x;
        return *this;
    }

    mat& operator*= (const U& x) {
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] *= x;
        return *this;
    }

    mat& operator/= (const U& x) {
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] /= x;
        return *this;
    }

    // Skalarni obicni operatori

    mat operator+ (const U& x) const {
        mat tmp = *this;
        tmp += x;
        return tmp;
    }

    mat operator- (const U& x) const {
        mat tmp = *this;
        tmp -= x;
        return tmp;
    }

    mat operator* (const U& x) const {
        mat tmp = *this;
        tmp *= x;
        return tmp;
    }

    mat operator/ (const U& x) const {
        mat tmp = *this;
        tmp /= x;
        return tmp;
    }

    // Matricni pointwise compound operatori
    mat& operator+= (const mat& x) {
        check_dims(x);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] += x.a[i][j];
        return *this;
    }

    mat& operator-= (const mat& x) {
        check_dims(x);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] -= x.a[i][j];
        return *this;
    }

    mat& operator*= (const mat& x) {
        check_dims(x);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] *= x.a[i][j];
        return *this;
    }

    mat& operator/= (const mat& x) {
        check_dims(x);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                a[i][j] /= x.a[i][j];
        return *this;
    }

    // Matricni obicni pointwise operatori
    mat operator+ (const mat& x) const {
        mat tmp = *this;
        tmp += x;
        return tmp;
    }

    mat operator- (const mat& x) const {
        mat tmp = *this;
        tmp -= x;
        return tmp;
    }

    mat operator* (const mat& x) const {
        mat tmp = *this;
        tmp *= x;
        return tmp;
    }

    mat operator/ (const mat& x) const {
        mat tmp = *this;
        tmp /= x;
        return tmp;
    }

    mat dot(const mat& x) const {
        if (empty() || x.empty())
            return mat();

        if (cols() != x.rows())
            throw "operand size mismatch";

        mat tmp(rows(), x.cols(), 0);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                for (size_t k=0; k<x.cols(); k++)
                    tmp[i][k] += a[i][j] * x[j][k];

        return tmp;
    }

    vec<U> dot(const vec<U>& x) const {
        if (empty() || x.empty())
            return vec<U>();

        if (cols() != x.size())
            throw "operand size mismatch";

        vec<U> tmp(rows(), (U)0);
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                tmp[i] += a[i][j] * x[j];

        return tmp;
    }

    mat T() const {
        if (empty())
            return mat();

        mat tmp(cols(), rows());
        for (size_t i=0; i<rows(); i++)
            for (size_t j=0; j<cols(); j++)
                tmp[j][i] = a[i][j];

        return tmp;
    }

    static mat id(size_t n) {
        mat t(n, n, 0);
        for (size_t i=0; i<n; i++)
            t[i][i] = 1;
        return t;
    }

    template<class V>
    friend std::ostream& operator<< (std::ostream& os, const mat<V>& v);

};

template<class T>
std::ostream& operator<< (std::ostream& os, const mat<T>& v) {
    if (v.empty())
        return os << "[]";

    os << "[" << v.a[0] << ",\n";
    for (size_t i=1; i<v.rows(); i++) {
        os << ' ' << v.a[i];
        if (i+1 != v.rows())
            os << ",\n";
        else
            os << "]";
    }
    return os;
}

typedef vec<float> vecf;
typedef vec<double> vecd;
typedef vec<long double> vecld;
typedef mat<float> matf;
typedef mat<double> matd;
typedef mat<long double> matld;


template<class T>
static T norm(vec<T> a) {
    T z = 0;
    for (T x : a) {
        z += x*x;
    }
    return sqrt(z);
}

}


#endif //PROJEKATC___LINEAR_ALGEBRA_H
