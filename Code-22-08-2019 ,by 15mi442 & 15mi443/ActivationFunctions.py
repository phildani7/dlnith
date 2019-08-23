import numpy as np

class ActivationFunctions:
    "Implementing Activation Functions"

    def sigmoid(x):
        ''' It returns 1/(1+exp(-x)).
            where the values lies between zero and one '''

        return 1/(1+np.exp(-x))

    def tanh(x):
        ''' It returns the value (1-exp(-2x))/(1+exp(-2x))
            and the value returned will be lies in between -1 to 1.'''

        return np.tanh(x)

    def arcTan(x):
        ''' It returns the value tanInverse(x)
            and the returned valus lies in between -1.570796327 to 1.570796327. '''

        return np.arctan(x)

    def ReLU(x):
        ''' It returns zero if the inputis less than zero
            otherwise it returns the given input. '''
        x1=[]
        for i in x:
            if i<0:
                x1.append(0)
            else:
                x1.append(i)

        return x1

    def leakyReLU(x):
        ''' If 'x' is the given input, then returns zero if the inputis less than zero
            otherwise it returns 0.01x . '''
        
        x1=[]
        for i in x:
            if i<0:
                x1.append(0.01*i)
            else:
                x1.append(i)
        #print(x,x1)

        return x1
    def softmax(x):
        ''' Compute softmax values for each sets of scores in x. '''
        return np.exp(x) / np.sum(np.exp(x), axis=0)



    

        

                



        
