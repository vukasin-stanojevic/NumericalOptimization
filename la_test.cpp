#include <bits/stdc++.h>
#include "la.h"
using namespace std;
using namespace la;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cerr.tie(nullptr);

	vecld a = {1, 2, 3}, b = {0, 10, 100};
	cout << a.dot(b) << '\n';
	cout << a << ", " << b << '\n';
	cout << (b/a) << '\n';
	cout << (b*a) << '\n';
	cout << (b-a) << '\n';
	cout << (b+a) << '\n';

	{
		auto t = a;
		a = b;
		b = t;
	}

	a = {0, 10, 100, 1000};
	cout << a << ", " << b << '\n';
	cout << matld::id(3).dot(a.outer(b).T()) << '\n';

	matf d = matf::id(10);
	d[9][8] = 1;

	cout << d;

	d = {{1, 2, 3}, {3, 4, 5}};

	cout << d;
	cout << d.dot(vecf{4, 1, 4});

}