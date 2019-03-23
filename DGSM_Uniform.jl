using Sobol
using ForwardDiff
using Statistics

#k number of variables
#N number of samples
#rl is row vector of size k consisting of lower bounds variables
#ru is row vector of size k consisting of upper bounds of variables


#the user needs to define the function as funct(x) outside the code
#and then call the function DGSM

#for example
#funct(x) = 5*x[1]+x[2]^2
#DGSM(2,400000,[2.0 5.0],[7.0 11.0])

#a dgsm with 2 variables 400000 samples and the range of x[1] is from 2 to 7 and range of x[2] is from 5 to 11

###output
#2×1 LinearAlgebra.Adjoint{Float64,Array{Float64,2}}:
#0.07654687726923803
#1.1816350479829216 

#the first numerical value is for x[1] and second one for x[2]

function DGSM(k , N, rl, ru)
    dfdx = zeros(Float64,N,k)
    Vu = zeros(Float64,1,k)
    DGSM = zeros(Float64,1,k)
    
    #sampling through QMC
    
    XX = zeros(Float64,N,k)
    s = SobolSeq(k)
    for i = 1:N
        x = next!(s)
        XX[i,:] = x
        
        for j = 1:k 
            index = i+(j-1)*N
            XX[index] = rl[j] + (ru[j] - rl[j])*XX[index]
        end
    end

    
    y0 = zeros(Float64,N,1)
    for i = 1:N
        y0[i] = funct(XX[i,:])
    end
    
    
    D = mean(y0.^2) - mean(y0)^2
    
    grad(x) = ForwardDiff.gradient(funct,x)
    
    for i =1:N
        temp = grad(XX[i,:])
        for j = 1:k
            dfdx[i+(j-1)*N] = temp[j]
        end
    end
    
    
    
    for j = 1:k
        Vu[j] = mean(dfdx[:,j].^2)
        DGSM[j] = Vu[j]*(ru[j]-rl[j]).^2/D/pi/pi
        
    end
    
    return DGSM'
end

#DGSM is column vector corresponding to DGSM of independent variables

            
        
