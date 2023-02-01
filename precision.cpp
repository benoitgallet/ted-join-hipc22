// C++ program to illustrate the
// floating-point error
#include<bits/stdc++.h>

using std::cout;
using std::endl;

#define TYPE double

TYPE floatError(TYPE no)
{
    TYPE sum = 0.0;
    for(int i = 0; i < 3; i++)
    {
        sum = sum + no;
    }
    return sum;
}

TYPE floatErrorDistance(int n)
{
    TYPE distance = 0.0;
    for (unsigned int i = 0; i < n; ++i)
    {
        distance += (0.001 - 0.0009) * (0.001 - 0.0009);
    }
    return distance;
}

// Driver code
int main()
{
    cout << std::setprecision(16);
    cout << floatError(0.001);
//    cout << floatErrorDistance(16);
}