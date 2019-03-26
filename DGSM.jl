using StatsFuns
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
    
    The next 2 columns of ith row i.e. 2nd and 3rd columns contains the information about the distribution
    Number || Column 2 || Column 3
    1 ====>   mean        standard deviation
    2 ====>   alpha       beta
    3 ====>   alpha       beta
    4 ====>   v1          v2 
    5 ====>   d           0.0
    6 ====>   0.0         0.0
    7 ====>   lambda      0.0
    
    These are the characterstics which define the distribution uniquely. A 0.0 in table represents that the 
    specific column data for the distribution is not required.
    For more information about the parameters being input for distributions refer the below link:
    https://juliastats.github.io/Distributions.jl/stable/univariate.html
    Eg-Suppose f(x) = sin(x[1])+x[2]^2
        x[1] is N(5,6) Distribution
        x[2] is Uniform Continous Random Distribution
        
        The distri matrix is
        1  5.0  6.0
        6  0.0  0.0


"""

mutable struct dgsm
    a::Float64
    absa::Float64
    asq::Float64
end

function element_generator(type_of_distribution, lower_limit, upper_limit, distribution_param1, distribution_param2)
    
    if(type_of_distribution == 1)
        while true
            
            x = rand(Normal(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 2)
        while true
            x = rand(Beta(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 3)
        while true
            x = rand(Gamma(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 4)
        while true
            x = rand(FDist(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 5)
        while true
            x = rand(TDist(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 6)
        x = lower_limit + (upper_limit - lower_limit)*rand()
        return x
    
    
    elseif(type_of_distribution == 7)
        while true
            x = rand(Poisson(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 8)
        while true
            x = rand(Arcsine(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 9)
        while true
            x = rand(BetaPrime(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 10)
        while true
            x = rand(Biweight(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 11)
        while true
            x = rand(Cauchy(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 12)
        while true
            x = rand(Chi(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 13)
        while true
            x = rand(Chisq(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 14)
        while true
            x = rand(Cosine(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 15)
        while true
            x = rand(Epanechnikov(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 16)
        while true
            x = rand(Erlang(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 17)
        while true
            x = rand(Exponential(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 18)
        while true
            x = rand(Frechet(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 19)
        while true
            x = rand(Gumbel(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 20)
        while true
            x = rand(InverseGamma(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 21)
        while true
            x = rand(InverseGaussian(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 22)
        while true
            x = rand(KSDist(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 23)
        while true
            x = rand(KSOneSided(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 24)
        while true
            x = rand(Laplace(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 25)
        while true
            x = rand(Levy(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 26)
        while true
            x = rand(Logistic(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 27)
        while true
            x = rand(LogNormal(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 28)
        while true
            x = rand(NoncentralChisq(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 29)
        while true
            x = rand(NoncentralT(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 30)
        while true
            x = rand(NormalCanon(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 31)
        while true
            x = rand(Pareto(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 32)
        while true
            x = rand(Rayleigh(distribution_param1))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 33)
        while true
            x = rand(SymTriangularDist(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 34)
        while true
            x = rand(Triweight(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 35)
        while true
            x = rand(VonMises(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    
    
    elseif(type_of_distribution == 36)
        while true
            x = rand(Weibull(distribution_param1,distribution_param2))
            if (x>= lower_limit && x<= upper_limit)
                return x
            end
        end
    end
    
end



function DGSM_Gen(f,k, samples, rangeLower, rangeUpper,distri)
    
    
    #Initialising the gradient matrix to zeros
    dfdx = zeros(Float64, samples, k)
    
    
    
    #XX is the matrix consisting of 'samples' number of sampling based on respective 
    #distributions of variables
    
    XX = zeros(Float64,samples,k)
    for i = 1:samples
        
        for j = 1:k
            index = i+(j-1)*samples
            
            XX[index] = element_generator(distri[j],rangeLower[j],rangeUpper[j],distri[j+k],distri[j+2*k])
            
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
    
    DGSM = [dgsm(0.0,0.0,0.0) for i in 1:k]
    
    for i = 1:k
        DGSM[i].a = mean(dfdx[:,i])
        DGSM[i].asq = mean(dfdx[:,i].^2)
    end
    
    
    dfdx = abs.(dfdx)
    
    #Evaluating E(|a|)
    
    for i = 1:k
        DGSM[i].absa = mean(dfdx[:,i])
    end
    
    #This function finally returns an array of structures, consisting a, absa and asq
    #respectively for the k independent parameters
    
    return DGSM
end
