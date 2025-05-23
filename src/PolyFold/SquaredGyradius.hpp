Real SquaredGyradius( cptr<Real> X, cref<Vector_T> mu )
{
    Real r2 = 0;
    
    for( Int i = 0; i < n; ++i )
    {
        r2 += SquaredDistance(Vector_T(X,i), mu);
    }
    
    r2 *= Frac<Real>(1,n);
    
    return r2;
}

Real SquaredGyradius( cptr<Real> X )
{
    if( recenterQ )
    {
        Real r2 = 0;
        
        for( Int i = 0; i < n; ++i )
        {
            r2 += Vector_T(X,i).SquaredNorm();
        }
        
        r2 *= Frac<Real>(1,n);
        
        return r2;
    }
    else
    {
        return SquaredGyradius( X, Barycenter(X) );
    }
}
