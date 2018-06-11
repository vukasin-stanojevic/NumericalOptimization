#ifndef _LAUTILS_H
#define _LAUTILS_H

#include "la.h"
#include <cmath>

namespace la {

template<class T>
T norm(vec<T> a) {
	T z = 0;
	for (T x : a)
		z += x*x;
	return sqrt(z);
}

} // namespace la

#endif