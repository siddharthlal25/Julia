using StatsFuns
using Sobol
using Distributions
using ForwardDiff
using Statistics

"""

The inputs to the function DGSM are as follows:
1.f: 
    This is the input function based on which the values of DGSM are to be evaluated
    Eg- f(x) = x[1]+x[2]^2
        This is function in 2 variables

2.k:
    Depicts the number of independent parameters in the function
    Eg- f(x) = log(x[1])+x[2]^2+x[3]
        => k = 3

3.samples:
    Depicts the number of sampling set of points to be used for evaluation of E(a), E(|a|) and E(a^2)
    a = partial derivative of f wrt x_i

4.rangeLower:
    This is a 1*k dimension matrix consisting of the lower bounds of the independent variables
    Eg- rangeLower = [x1(l) x2(l) x3(l).....xk(l)]
        consists the lower bounds of the corresponding variable in the above displayed manner

5.rangeUpper:
    This is a 1*k dimension matrix consisting of the upper bounds of the independent variables
    Eg- rangeUpper = [x1(u) x2(u) x3(u).....xk(u)]
        consists the upper bounds of the corresponding variable in the above displayed manner

6.distri:
    This is a k*3 dimension matrix depicting the distribution of the k independent variables 
    The first column of the ith row depicts the distribution of the x_i 
    Number || Type
    1 ===>    Normal Distribution
    2 ===>    Beta Distribution
    3 ===>    Gamma Distribution
    4 ===>    F Distribution
    5 ===>    T Distribution
    6 ===>    Uniform Continous Distribution
    7 ===>    Poisson Distribution
    8 ===>    Binomial Distribution
    9 ===>    Negative Binomial Distribution

    The next 2 columns of ith row i.e. 2nd and 3rd columns contains the information about the distribution
    Number || Column 2 || Column 3
    1 ====>   mean        standard deviation
    2 ====>   alpha       beta
    3 ====>   alpha       beta
    4 ====>   v1          v2 
    5 ====>   d           0.0
    6 ====>   0.0         0.0
    7 ====>   lambda      0.0
    8 ====>   n           p
    9 ====>   r           p

    These are the characterstics which define the distribution uniquely. A 0.0 in table represents that the 
    specific column data for the distribution is not required.

    For more information about the parameters being input for distributions refer the below link:
    https://juliastats.github.io/Distributions.jl/stable/univariate.html

    Eg-Suppose f(x) = sin(x[1])+x[2]^2+x[3]
        x[1] is N(5,6) Distribution
        x[2] is Uniform Continous Random Distribution
        x[3] is T Distribution with d = 4.2
        The distri matrix is
        1  5.0  6.0
        6  0.0  0.0
        5  4.2  0.0
"""


function DGSM(f,k, samples, rangeLower, rangeUpper,distri)
    
    #Initialising the limits to zero
    sobolLimitsUpper = zeros(Float64,1,k)
    sobolLimitsLower = zeros(Float64,1,k)
    
    #Initialising the gradient matrix to zeros
    dfdx = zeros(Float64, samples, k)
    
    #Initialising the E(|a|) E(a) and E(a^2) matrix to zeros
    #a is partial derivative of f wrt to xi
    a = zeros(Float64,k,1)
    absa = zeros(Float64,k,1)
    asq = zeros(Float64,k,1)
    
    
    #Determining the upper and lower limits of k-dimensional sobol sequences
    #Transforming the real bounds into Sobol sequence bounds using cdf
    for i = 1 : k
        
        if(distri[i]==1)
            sobolLimitsUpper[i] = normcdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = normcdf(distri[i+k],distri[i+2*k],rangeLower[i])
            
        end
        
        if(distri[i]==2)
            sobolLimitsUpper[i] = betacdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = betacdf(distri[i+k],distri[i+2*k],rangeLower[i])
            
        end
        
        if(distri[i]==3)
            sobolLimitsUpper[i] = gammacdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = gammacdf(distri[i+k],distri[i+2*k],rangeLower[i])
        
        end
        
        if(distri[i]==4)
            sobolLimitsUpper[i] = fdistcdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = fdistcdf(distri[i+k],distri[i+2*k],rangeLower[i])
        
        end
        
        if(distri[i]==5)
            sobolLimitsUpper[i] = tdistcdf(distri[i+k],rangeUpper[i])
            sobolLimitsLower[i] = tdistcdf(distri[i+k],rangeLower[i])
        
        end
        
        if(distri[i]==6)
            sobolLimitsUpper[i] = 1.0
            sobolLimitsLower[i] = 0.0
        
        end
        
        if(distri[i] == 7)
            sobolLimitsUpper[i] = poiscdf(distri[i+k],rangeUpper[i])
            sobolLimitsLower[i] = poiscdf(distri[i+k],rangeLower[i])
        
        end
        
        if(distri[i]==8)
            sobolLimitsUpper[i] = binomcdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = binomcdf(distri[i+k],distri[i+2*k],rangeLower[i])
        
        end
        
        if(distri[i]==9)
            sobolLimitsUpper[i] = nbinomcdf(distri[i+k],distri[i+2*k],rangeUpper[i])
            sobolLimitsLower[i] = nbinomcdf(distri[i+k],distri[i+2*k],rangeLower[i])
        
        end
        
    end
    
    
    #s is assigned the upper and lower limits of the bounds of sobol limits
    s = SobolSeq(sobolLimitsLower, sobolLimitsUpper)
    
    
    #XX is the matrix consisting of 'samples' number of sampling based on respective 
    #distributions of variables
    
    XX = zeros(Float64,samples,k)
    for i = 1:samples
        x = next!(s)
        XX[i,:] = x
        
        
        for j = 1:k 
            index = i+(j-1)*samples
            
            if(distri[j] == 1)
                XX[index] = norminvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
            if(distri[j] == 2)
                XX[index] = betainvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
            if(distri[j] == 3)
                XX[index] = gammainvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
            if(distri[j] == 4)
                XX[index] = fdistinvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
            if(distri[j] == 5)
                XX[index] = tdistinvcdf(distri[j+k],XX[index])
            end
            
            if(distri[j] == 6)
                XX[index] = rangeLower[j] + (rangeUpper[j]-rangeLower[j])*XX[index]
            end
            
            if(distri[j] == 7)
                XX[index] = poisinvcdf(distri[j+k],XX[index])
            end
            
            if(distri[j] == 8)
                XX[index] = binominvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
            if(distri[j] == 9)
                XX[index] = nbinominvcdf(distri[j+k],distri[j+2*k],XX[index])
            end
            
        end
    end
    
    #function to evaluate gradient of f wrt x
    grad(x)= ForwardDiff.gradient(f,x)
    
    #Evaluating the derivatives with AD
    for i =1:samples
        temp = grad(XX[i,:])
        for j = 1:k
            dfdx[i+(j-1)*samples] = temp[j]
        end
    end
    
    #Evaluating E(a) and E(a^2)
    
    a = [mean(dfdx[:,x]) for x in 1:k]
    asq = [mean(dfdx[:,x].^2) for x in 1:k]
    
    dfdx = abs.(dfdx)
    
    #Evaluating E(|a|)
    
    absa = [mean(dfdx[:,x]) for x in 1:k]
    
    #This function finally returns a matrix ok k*3 matrix, consisting E(a), E(|a|) and E(a^2)
    #respectively for the k independent parameters
    #The ith row consists of E(a), E(|a|) and E(a^2) for the ith independent parameter
    
    outputf = hcat(a, absa)
    outputf = hcat(outputf , asq)
    return outputf
end
    
