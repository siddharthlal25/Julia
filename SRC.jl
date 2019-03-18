#This code primarily focuses on calculating the SRC's for given input by the user
#The user has to input a column vector of dependant variable 
#The user has to input k that is the number of independant variables
#The number of samples(n) provided is extracted from length of the dependant variable Y vector
#Therefore Y is n*1 column vector and X is n*k matrix
#The output would be a column vector with src's of first variable in first place followed by second variable
#and so on...

using StatsBase

#Function for evaluating the standard deviation of a data stored in a vector with known mean

function std_dev(x,mean_x)
    sum =0
    len = length(x)
    for i in x
        sum += (i - mean_x)^2
    end
    sum = sum/len
    sqrt(sum)
    return sum
end

function disp(x)
       row = size(x,1)
       col = size(x,2)
       for i = 1:row
           for j = 1:col
               index = i + (j-1)*row
               num = x[index]
               print("$num ")
           end
           println()
       end       
end

        

#x is a matrix with n rows and k columns
#Y is a row vector with n rows

function SRC(y,x,k)
    n = length(y)
    mean_y = mean(y)
    stdm_y = std_dev(y,mean_y)

    #######
    #Normalising the y values
    for i = 1:length(y)
        y[i] -= mean_y
        y[i] /= stdm_y
    end
    #######
    mean_x = zeros(Float64, 1, k)
    for i = 1:k
        mean_x[i] = mean(x[:,i])
    end
    
    #######
    #Normalising the x values
    for i = 1:k
        stdm_x = std_dev(x[:,i],mean_x[i])
        for j = 1:n
            index = j+(i-1)*n
            x[index] -= mean_x[i] 
            x[index] = x[index]/stdm_x
        end
    end
    #######
    
    ######
    #Evaluating the coefficients of linear regression using least square method
    beta = inv(x' * x)*x' * y
    ######
    
    return beta
    
end    
    
function main()
    println("Enter the number of independant parameters:")
    k = parse(Int64,readline())
    println()
    println("Enter the number of sample points:")
    n = parse(Int64,readline())
    println()
    y = zeros(Float64,n,1)
    x = zeros(Float64,n,k)
    for i = 1:n
        println("Sample$i:")
        println()
        println("Enter y:")
        y[i] = parse(Float64,readline())
        println()
        for j=1:k
            print("Enter x$j:")
            x[i+(j-1)*n] = parse(Float64,readline())
            println()
        end
    end
    println()
    println("Generated Y matrix:")
    disp(y)
    println()
    println()
    println("Generated X matrix:")
    disp(x)
    println()
    println()
    println("The values of SRC's are as follows:")
    println("The first value for first independant parameter and second for second and so on...")
    betas = SRC(y,x,k)
    disp(betas)
    println()
end

main()
            
