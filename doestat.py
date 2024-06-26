import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns             #library for analysis

from scipy.stats import t         #library for regression

from scipy.stats import f
from scipy.stats import linregress
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, Latex
import sys                        #library for surface graph


from sympy import symbols         #sympy library for symbols, diff, solve and subs

class analysis: 
    
    """
    Class -> analysis(X,y,effect_error,t) - Class for calculating factorial planning effects.
    Instantiate this class to access the following methods: probability_graph(), effect_percentages(), analysis_effect().

    Attributes

    X: matrix containing the effects to be calculated.
    y: vector containing the response.
    effect_error: error of an effect. It will be 0 if no replicas are made.
    t: t value from the t-Student distribution.
    
    Methods

    analysis_effect: returns "Probability" and "Effect Percentages" graphs, and Excel tables with generated data.

    Acknowledgements: Prof.Dr.Edenir Pereira Filho and B.S. André Simão
    
    """
    def __init__(self, x, y, effect_error=0, t=0):
        self.X = x
        self.y = y
        self.effect_error = effect_error
        self.t = t
        self.start = [0]
        self.center = []
        self.end = []
        self.gauss = [] 
         
    @property
    def __matrix_x(self):
        return self.X
    
    @property
    def vector_y(self):
        return self.y

    
    @property
    def effect(self):  # Returns product values between effects and response
        return (self.X.T * self.y).T

    
    @property
    def __n_effect(self):  # Returns dimensions of the matrix with effects (coded_value * response)
        return self.X.shape

    
    @property
    def __effect_indices(self):  # Returns list with respective interactions
        return self.X.T.index


    
    @property
    def __generate_start_center_end_gauss(self):  # Returns the values of the Gaussian
        for i in range(self.__n_effect[1]):
            self.end.append(self.start[i] + (1 / self.__n_effect[1]))
            self.start.append(self.end[i])
            self.center.append((self.start[i] + self.end[i]) / 2)
            self.gauss.append(norm.ppf(self.center))
        return self.gauss

    
    def __calculate_effects(self):  # Returns vector with effects
        return (np.einsum('ij->j', self.effect)) / (self.__n_effect[0] / 2)  # np.einsum -> function that sums columns of a matrix

    def __calculate_percentage_effects(self):  # Returns vector with probability
        return (self.__calculate_effects() ** 2 / np.sum(self.__calculate_effects() ** 2)) * 100

    def __define_gaussian(self):  # Returns the values of the Gaussian
        return self.__generate_start_center_end_gauss[self.__n_effect[1] - 1]

    def __label(self, axs):  # Marks points on the probability graph
        for i, label in enumerate(self.__sort_effects_probabilities().index):
            axs[0].annotate(label, (self.__sort_effects_probabilities()['Effects'].values[i],
                                    self.__define_gaussian()[i]))

    def __sort_effects_probabilities(self):  # Returns dataframe sorted in ascending order with effect values
        data = pd.DataFrame({'Effects': self.__calculate_effects()}, index=self.__effect_indices)
        data = data.sort_values('Effects', ascending=True)
        return data

    def __define_ic(self):  # Returns set of IC points
        return np.full(len(self.__define_gaussian()), self.effect_error * self.t)

    def __check_ic(self, axs):
        if self.effect_error == 0 or self.t == 0:
            pass
        else:
            return self.__plot_ic(axs)

    def __graphics_analysis(self):
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        
        axs[0].scatter(self.__sort_effects_probabilities()['Effects'],
                       self.__define_gaussian(), s=40, color='darkred')
        axs[0].set_title('Probability Graph', fontsize=18, fontweight='black', loc='left')
        axs[0].set_ylabel('z')
        axs[0].set_xlabel('Effects')
        self.__label(axs)
        self.__check_ic(axs)
        axs[0].grid(color='k', linestyle='solid')
        
        sns.set_style("whitegrid")
        sns.load_dataset("tips")
        
        axs[1] = sns.barplot(x='Effects', y='%', color='purple', data=pd.DataFrame(
            {'Effects': self.__effect_indices, '%': self.__calculate_percentage_effects()}))
        axs[1].set_title('Effect Percentages', fontsize=16, fontweight='black', loc='left')
        
        fig.suptitle('Effect Graphs Analysis', fontsize=22, y=0.99, fontweight='black', color='darkred')
        plt.tight_layout()
        plt.savefig('fabi_effect_graphs.png', transparent=True)
        
    def __plot_ic(self, axs):  
        axs[0].plot(-self.__define_ic(), self.__define_gaussian(), color='red')
        axs[0].plot(0 * self.__define_ic(), self.__define_gaussian(), color='blue')
        axs[0].plot(self.__define_ic(), self.__define_gaussian(), color='red')

    def analysis_effect(self):
        """
        Function -> analysis_effect
        Function to calculate factorial planning effect
           
        Parameters
        -----------
        
        X = matrix containing the effects to be calculated
        y = vector containing the response
        effect_error = error of an effect. It will be 0 if no replicas are made
        t = t value corresponding to the degrees of freedom of the error of an effect. It will be 0 if no replicas are made.
        
        Returns
        -----------
        
        Graphs of "Percentage x Effects" (barplot) and "Probability" (scatter) from the fabi_effect routine in Octave
        
        """
        return plt.show(self.__graphics_analysis())

    
class CP:
    """
    Class -> CP(y, k) - Class for calculating value-t and effect_error 
        
    Atributes
    -----------
    y: pd.Series - values of center points region
    
    k: int -  number of variables 
    
    Methods
    -----------
    invt: return t-value.
    
    effect_error: return the error of one efffect
    
    pqes: return Pure Quadratic Error Sum
    
    df_pqes: return the degrees of freedom
   
    """
    def __init__(self,y=None , k=None):
        self.y = y
        self.k = k

        
    def __array(self): 
        return self.y.values
    
    def __error_exp(self):
        return self.y.std()
    
    def __df(self):
        """Calculating value-t of t-Student bimodal distribution"""
        return self.y.shape[0]-1
    
    def __check_df(self):
        return 
    
    def invt(self, df_a = None):
        """
        return value-t of t-Student bimodal distribution.
        
        Parameters
        -----------
        
        (optional) df_a: degree of freedom that not belong to CP class
        
        Returns:
        
        t-value type float
        
        """
        if (df_a == None):
            return t.ppf(1-.05/2,self.__df())
        else:
            return t.ppf(1-.05/2,df_a)
        
    def __message_error_11(self):
        return print('Erro11: Invalid parameters.')
    
    def __calculate_effect_error(self): 
        return 2*self.__error_exp()/(self.y.shape[0]*2**self.k)**0.5
    
    def effect_error(self):
        """Return effect_error value"""
        if self.k == None or self.y.all() == None:
            return self.__message_error_11()
        else:
            return self.__calculate_effect_error()
    
    def __calculate_pqes(self):
     
        return np.sum((self.__array() - np.mean(self.__array()))**2)
    
    def pqes(self):     
        """Return pqes value"""
        if self.y.all() == None:
            return self.__message_error_11()
        else:
            return self.__calculate_pqes()
    
    def  df_pqes(self):
        """Return degree of freedom"""
        return len(self.y)

class regression_analysis:
    """
        Class -> regression(X, y, pqes, dof) - Create a regression model and adjust it through Variance Analysis
        
        This routine aims to calculate regression models using the following equation:
        
        $inv(X^tX)X^ty$
        
        Attributes
        -------------------
        X: Matrix with the coefficients to be calculated (type: pandas.DataFrame)
        
        y: Response that will be modeled (type: pandas.Series)
        
        pqes (optional): Pure Error Sum of Squares of the values at the Central point (type: float or int)
        Use pde.CP(yc).pqes() to calculate
        For more info: help(pde.CP.pqes)
        
        dof (optional): Degrees of freedom of the central point (type: int)
        Use pde.CP(yc,k).dof_pqes()
        For more info: help(pde.CP.df_pqes)
        
        NOTE! THIS FEATURE IS STILL IN DEVELOPMENT AND DOES NOT FUNCTION WHEN THERE ARE DATA REPLICATES!
        
        auto (optional): Automate the exclusion of significant coefficients (type: bool)
        For more info: help(pde.regression.auto)
        
        self_check (optional): Automate the check for lack of model fit through analysis of variance.
        For more info: help(pde.regression.self_check)
        
        Methods
        --------------------
        create_table_anova: Returns the ANOVA table of the created model (type: NoneType)
        For more info: help(pde.regression.create_table_anova)
        
        plot_graphs_anova: Returns graphs with the parameters of the ANOVA Table (type: NoneType)
        For more info: help(pde.regression.plot_graphs_anova)
        
        plot_graphs_regression: Returns graphs of the regression model (type: NoneType)
        For more info: help(pde.regression.plot_graphs_regression)
        
        model_coefficients: Returns a list with the model coefficients, with insignificant coefficients having null value.
        
        recalculate_coefs: Returns a pandas.DataFrame with the significant model coefficients
        
        regression_results: Master function that creates a regression model and adjusts it through Analysis of Variance
        For more info: help(pde.regression.regression_results)
        
    """

   
    __check_ci = True    
    __final_msg = '\033[1mAnalysis complete! Check the results on your directory.'
    
    def __init__(self, X:object, y:object, SSPE=None, df=None, auto=False, self_check=False):
        self._X = X
        self.y = y
        self.SSPE = SSPE
        self.df = df 
        self._auto = auto
        self._self_check = self_check
            
    def __n_exp(self):
        return  self.X.shape[0]
    
    @property
    def self_check(self):
        return self._self_check
    
    @self_check.setter
    def self_check(self, value:bool) -> bool:
        if isinstance(value,bool):
            self._self_check = value
            
    @property
    def auto(self):
        return self._auto
    
    @auto.setter # changes auto False to True for exclude columns insignificants after regression_results function
    def auto(self, value:bool) -> bool:
        if isinstance(value,bool):
            self._auto = value
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, new_dataframe:object) -> object:
        if isinstance(new_dataframe,object):
            self._X = new_dataframe
        
    def __n_coef(self):
        return self.X.shape[1]
    
    def __matrix_X(self):
        return self._X.values 
    
    def __array_y(self):
        return self.y.values
    
    def __calculate_var_coefs(self):
        """
        Return coeficients variance values
        
        Eq.: diag(inv(X'*X))
        """
        return np.diagonal(np.linalg.inv(np.matmul(self.__matrix_X().T,self.__matrix_X()))).round(3)
    
    def __calculate_matrix_coef(self):
        """
        Return an array with the results of the equation below:
        
        b = inv(X'*X))*(X'*Y)
        """
        return np.matmul(np.linalg.inv(np.matmul(self.__matrix_X().T,self.__matrix_X())),
                         self.__matrix_X().T*self.__array_y()).T
    
    def calculate_coefs(self):
        """Return the sum of restult of the definition: "__matrix_coef" """
        return np.einsum('ij->j', self.__calculate_matrix_coef()).round(5)
    
    
    def __calculate_pred_values(self):
        """Retunr the values of predict by the model"""
        return np.matmul(self.X,self.calculate_coefs())
    
 
    def predict(self, value=0):
        """Retunr the values of predict by the model"""
        return np.matmul(self.X,self.calculate_coefs()+value)
    
    def __calculate_residuals(self):
        """Retunr the residuals values of predict by the model"""
        return self.__array_y()-self.__calculate_pred_values()
    
    # Sum of Squares - Part 1
    
    def __calculate_SSreg(self):
        return np.sum((self. __calculate_pred_values()-self.__array_y().mean())**2)
    
    def __calculate_SSres(self):
        return np.sum(self.__calculate_residuals()**2)

    def __calculate_SSTot(self):
        return np.sum(self.__calculate_SSreg()+self.__calculate_SSres())
    
    def __calculate_SSLoF(self):
        return self.__calculate_SSres()-self.SSPE
    
    def __calculate_R2(self):  
        return self.__calculate_SSreg()/self.__calculate_SSTot()
        
    def __calculate_R2_max(self):
        return (self.__calculate_SSTot()-self.SSPE)/self.__calculate_SSTot()
        
    def __calculate_R(self):
        return self.__calculate_R2()**.5
    
    def __calculate_R_max(self):
        return self.__calculate_R2_max()**0.5
    

    # Sum of Squares - Part 2 (deggres of freedom)
    
    def __df_SSreg(self):
        return self.__n_coef()-1
    
    def __df_SSres(self):
        return self.__n_exp()-self.__n_coef()
    
    def __df_SSTot(self):
        return self.__n_exp()-1
    
    def __df_SSLof(self):
        return (self.__n_exp() - self.__n_coef()) - self.df
    
    # Mean of Squares - Part 3
    
    def __calculate_MSreg(self):
        return self.__calculate_SSreg()/self.__df_SSreg()
    
    def __calculate_MSres(self):
        return self.__calculate_SSres()/self.__df_SSres()
    
    def __calculate_MSTot(self):
        return self.__calculate_SSTot()/self.__df_SSTot()
    
    def __calculate_MSPE(self):
        return self.SSPE/self.df
    
    def __calculate_MSLoF(self):
        if self.__df_SSLof() != 0:
            return self.__calculate_SSLoF()/self.__df_SSLof()
        else:
            return self.__calculate_SSLoF()
    
    # F Tests
    
    def __ftest1(self):
        return self.__calculate_MSreg()/self.__calculate_MSres()
    
    def __ftest2(self):
        return self.__calculate_MSLoF()/self.__calculate_MSPE()
    
    # F table
    
    def __ftable1(self): 
        return f.ppf(.95, self.__df_SSreg(),self.__df_SSres()) #F1 with 95% of confidence
    
    def __ftable2(self): 
        return f.ppf(.95, self.__df_SSLof(),self.df) #F1 with 95% of confidence
    
    # ANOVA Table
    def __anova_list(self):
        """ANOVA table structure"""
        return [
        ['\033[1m'+'Parameter','Sum of Squares (SS)','Degrees of Freedom (DoF)','Mean Square (MS)','F1-Test'+'\033[0m'],
        ['\033[1mRegression:\033[0m','%.0f'%self.__calculate_SSreg(),self.__df_SSreg(),'%.0f'%self.__calculate_MSreg(),'%.1f'%self.__ftest1() ],
        ['\033[1mResidual:\033[0m', '%.1f'%self.__calculate_SSres().round(2), self.__df_SSres(),'%.2f'%self.__calculate_MSres(),'%.1f'%self.__ftest1()],
        ['\033[1mTotal:\033[0m', '%.0f'%self.__calculate_SSTot(), self.__df_SSTot(), '%.0f'%self.__calculate_MSTot(), '\033[1mF2-Test\033[0m'],
        ['\033[1mPure Error:\033[0m','%.2f'%self.SSPE, self.df, '%.2f'%self.__calculate_MSPE(), '%.2f'%self.__ftest2() ],
        ['\033[1mLack of Fit:\033[0m', '%.2f'%self.__calculate_SSLoF(), self.__df_SSLof(), '%.2f'%self.__calculate_MSLoF(), '\033[1mF Tabulated\033[0m'],
        ['\033[1mR²:\033[0m', '%.4f'%self.__calculate_R2(), '\033[1mR:\033[0m', '%.4f'%self.__calculate_R(),'F1: %.3f'%self.__ftable1() ],
        ['\033[1mR² max:\033[0m','%.4f'%self.__calculate_R2_max(), '\033[1mR max:\033[0m', '%.4f'%self.__calculate_R_max(),'F2: %.3f'%self.__ftable2()]
        ]
        
    def create_table_anova(self,show=False):
        """Return Nonetype with ANOVA table"""
        if show == True:
            return tabulate(self.__anova_list())
        else:
            print('{:^110}'.format('\033[1m'+'ANOVA Table'+'\033[0m'))
            print('-='*53)
            print(tabulate(self.__anova_list(),tablefmt="grid"))
            print('-='*53)

    #Data visualization 
    
    def plot_graphs_anova(self):
        """
        Return the graph visualization of ANOVA parameters 
                
        Returns
        ---------
        1 - Mean Squares graph : 
        
            - MS of regression
            - MS of residuals and the respective t-Student values
            - MS of pure error
            - MS of lack of fit and the respective t-Student values
        
        2 - F2-test graph - MSLof/MSPE:
        
            - F2 Value 
            - Tabulated F Value
            - Ratio of F2/Tabulated F
        
        3 - F1-test graph - MSReg/MSRes:
        
            - F1 Value
            - Tabulated F Value
            - Ratio of F1/Tabulated F
            
        4 - Coefficient of Determination graph:
            
            - Explained Variation 
            - Maximum Explained Variation
        """
        fig = plt.figure(constrained_layout=True,figsize=(10,10))          
        subfigs = fig.subfigures(2, 2, wspace=0.07, width_ratios=[1.4, 1.])

        #Mean of Squares 
        axs0 = subfigs[0,0].subplots(2, 2)

        axs0[0,0].bar('MSReg',self.__calculate_MSreg(),color='darkgreen' ,)
        axs0[0,0].set_title('MQ Regression',fontweight='black')
        axs0[0,0].text(-.35, 200, '%.1f'%self.__calculate_MSreg(), fontsize=20,color='white')

        axs0[0,1].bar('MSRes e t',self.__calculate_MSres(),color='darkorange')
        axs0[0,1].set_title('MQ Residual',fontweight='black')
        axs0[0,1].text(-.35,.5, '%.1f  %.3f'%(self.__calculate_MSres(),CP().invt(self.__df_SSres()-1)), fontsize=20,color='k')
        #axs0[0,1].text(-.35, 2.07, '%.4f'%CP().invt(self.__df_SSres()-1), fontsize=20,color='k')

        axs0[1,0].bar('MSPE',3, color= 'darkred')
        axs0[1,0].set_title('MS of Pure Error',fontweight='black')
        axs0[1,0].text(-.35, 1.27,'%.2f'%self.__calculate_MSPE(), fontsize=20,color='w')

        axs0[1,1].bar('MSLoF e t',3,color= 'darkviolet')
        axs0[1,1].set_title('MS of Lack of Fit',fontweight='black')
        axs0[1,1].text(-.35, 1.98, '%.1f'%self.__calculate_MSLoF(), fontsize=20,color='w')
        axs0[1,1].text(-.35, 1.07, '%.4f'%CP().invt(self.__df_SSLof()), fontsize=20,color='w')

        
        #F2-test
        axs1 = subfigs[0,1].subplots(1, 3)

        axs1[0].bar('MSLof/MSPE',self.__ftest2(),color='darkred' ,)
        axs1[0].set_title('F2-test',fontweight='black')

        axs1[1].bar('F2',self.__ftable2(),color='darkred')
        axs1[1].set_title('F2 tabulated',fontweight='black')

        axs1[2].bar('F2calc/ Ftab',self.__ftest2()/self.__ftable2(), color= 'darkred')
        axs1[2].set_title(r'$\bf\frac{F2_{calculated}}{F2_{tabulated}}$',fontweight='black',fontsize=16,y=1.031)
        axs1[2].axhline(1,color='black')

        #F1 tests (testes F)
        axs2 = subfigs[1,0].subplots(1, 3)

        axs2[0].bar('MSReg/MSRes',self.__ftest1(),color='navy' ,)
        axs2[0].set_title('F1-test',fontweight='black')

        axs2[1].bar('F1',self.__ftable1(),color='navy')
        axs2[1].set_title('F1-tabulated',fontweight='black')

        axs2[2].bar('F1calc/ Ftable',self.__ftest1()/self.__ftable1(), color= 'navy')
        axs2[2].set_title(r'$\bf\frac{F1_{calculated}}{F1_{tabulated}}$',fontweight='black',fontsize=16,y=1.031)
        axs2[2].axhline(10,color='w')
        
        #Coefficient of Determination 
        axs3 = subfigs[1,1].subplots(1, 2)
        axs3[0].bar('R²',self.__calculate_R2(),color='dimgray' ,)
        axs3[0].set_title('Explained Variation',fontweight='black')
        axs3[0].axhline(1,color='k')
        
        axs3[1].bar('R² max',self.__calculate_R2_max(),color='dimgray')
        axs3[1].set_title('Explained Variation \n Maximum',fontweight='black')
        axs3[1].axhline(1,color='k')
        
     
        fig.suptitle('ANOVA (Analisys of Variance)', fontsize=20, fontweight='black',y=1.05)
        plt.savefig('ANOVA (Analisys of Variance).png',transparent=True)

      
        return plt.show()

    
    #  Verification of regression coefficients

    
    def  __user_message(self):
        return input('\n\n'+'\033[1mDoes the model have lack of fit? [Y/N]  \033[0m'+'\n\n')
    
    def __check_model(self): #Return boolean variable for define confidence interval through user message
        check_answer = self.__user_message().upper()
        if check_answer == 'Y':
            return True
        elif check_answer == 'N': # this change will be importan in recalculate_model function for decide confidence interval
            return False
        else:
            print('\033[1mError: only "Y" or "N" responsed will be accepted.')
            print('Finish!')
            return sys.exit()
        
    def __self_turning(self, msg=False):
        if (self.__ftest1() > self.__ftable1()) or (self.__ftest2() < self.__ftable2()):
            if msg == True:
                display(Latex(f'$$The\;model\;does\;not\;have\;lack\;of\;fit$$'))
            return False
        else:
            if msg == True:
                display(Latex(f'$$The\;model\;have\;lack\;of\;fit$$'))
            return True
    
    
    def define_ic_coefs(self,msg=False): #decides if will calculate ic mslof or msres
        if self.self_check == False:
            check_answer = self.__check_model() #Returns True or False through this method
        elif self.self_check == True:
            check_answer = self.__self_turning(msg)
        else:
            raise TypeError('"doe.regression_analysis().resgression_results()" one of argument is missing for "self.check".')
           
        if check_answer == True:
            regression_analysis.__check_ci = True # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSLoF() # to calculate interval confidence for lack of fit
        elif check_answer == False:
            regression_analysis.__check_ci = False # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSRes() #to calculate interval confidence for residues 
        
    def show_ci(self, manual=None):
        """Absolute values of model's confident interval"""
        if regression_analysis.__check_ci == True or manual ==True:
            return self.__define_ic_MSLoF()
        elif  regression_analysis.__check_ci == False or manual == False:
            return self.__define_ic_MSRes()
    
    def __define_ic_MSLoF(self): #calculates confidence interval for mslof
        return (((self.__calculate_MSLoF()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSLof()-1)).round(4)
        
    def __define_ic_MSRes(self): #calculates confidence interval for msresN
        return (((self.__calculate_MSres()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSres()-1)).round(4)
    
    def plot_graphs_regression(self):
        """
        Return the regression model graph for anlysis of insignificant variables to the model.
        
        Returns
        --------
        
        1 - Experimental values versus Predict and the respectives confident intervals.
        
        2 - Predict versus residual.
        
        3 - Histogram of residual.
        
        4 - Regression coefficients and respectives confident intervals.
        
        
        """
        # fig = plt.figure(constrained_layout=True,figsize=(10,10))          #This work in linux/ubuntu
        # subfigs = fig.subfigures(2, 2, wspace=0.07, width_ratios=[1.4, 1.])

        fig = plt.figure(figsize=(10,14)) # This work better in windows
        spec = fig.add_gridspec(3, 2)
        fig.tight_layout()
      
        axs0 =  fig.add_subplot(spec[0, :])
        
        m, b, r_value, p_value, std_err = linregress(self.y, self.__calculate_pred_values())
        
        axs0.plot(self.y, m*self.y + b,color='darkred')
        axs0.legend(['y = {0:.3f}x + {1:.3f}'.format(m,b) +'\n'+'R= {0:.4f}'.format((r_value)**.5)])
        axs0.scatter(self.y,self.predict(),color='b',marker=">",s=40)
        #axs0.scatter(self.y,self.predict(-self.show_ci()),color='b',marker="+",s=20)
        axs0.set_title('Experimental x Predict',fontweight='black')
        axs0.set_ylabel('Predict')
        axs0.set_xlabel('Experimental')
        axs0.grid()
        
        axs1 =  fig.add_subplot(spec[1, 0])
        
        axs1.scatter(x=self.__calculate_pred_values(), y=self.__calculate_residuals(),marker="s",color='r')
        axs1.set_title('Predict x Residual',fontweight='black')
        axs1.set_xlabel('Predict')
        axs1.set_ylabel('Residual')
        axs1.axhline(0,color='darkred')
        axs1.grid()
        
        axs2 = fig.add_subplot(spec[1, 1])
        
        axs2.hist(self.__calculate_residuals(),color ='indigo',bins=30)
        axs2.set_title('Histogram of the Residual',fontweight='black')
        axs2.set_ylabel('Frequence')
        axs2.set_xlabel('Residual')
        
        axs3 =  fig.add_subplot(spec[2, :])
        
        axs3.errorbar(self.X.columns,self.calculate_coefs(),self.define_ic_coefs(True), fmt='^', linewidth=2, capsize=6, color='darkred')
        axs3.axhline(0,color='darkred', linestyle='dashed')
        axs3.set_ylabel('Coefficient Values')
        axs3.set_xlabel('Coefficient')
        axs3.set_title('Regression of Coefficients',fontweight='black')
        axs3.grid()
        
        fig.suptitle('Regression Model'+'\n' + '-- regression_analysis --', fontsize=20, fontweight='black',y=1.1)
        plt.savefig('Regression Model.png',transparent=True)
        
        return plt.show()
    
    
    # Recalculate the model and to variables insignificant excludes automatically    
    
    def dict_coefs_ci(self): #list with dicts {'coefs':values,'coefs_max':values,'coefs_min':values}
        return  [dict(zip(self.X.columns, self.calculate_coefs())),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coefs()[0].round(4))),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coefs()[1].round(4)))]
    
    def recalculate_coefs(self):  #returns an array with coefs values and coefs insignificants equal zero 
        """Return a DataFrame with significants coefficients"""
        return self.__delete_coefs_insignificants_matrix()
       
    def __calculate_inter_max_min_coefs(self): #returns a tuple with (coef+ci,coef-ci)
        if regression_analysis.__check_ci == True:
            return [self.calculate_coefs() + self.__define_ic_MSLoF(), 
                            self.calculate_coefs() - self.__define_ic_MSLoF()]
        elif  regression_analysis.__check_ci == False:
            return [self.calculate_coefs() + self.__define_ic_MSRes(),
                            self.calculate_coefs() - self.__define_ic_MSRes()] 
    
        
    def __delete_coefs_insignificants(self): #select (coef - ci <= coef <= coef + ci) and replace for zero
        coefs = self.dict_coefs_ci()[0]
        max_ = self.dict_coefs_ci()[1]
        min_ = self.dict_coefs_ci()[2]
        for coef in coefs.keys():
            if min_[coef]<= 0 <= max_[coef]:
                coefs[coef] = 0
        return coefs
    
    def model_coefients(self):
        """Return a values list with significants coefficients and null values for insignificants coefficients"""
        return list(self.__delete_coefs_insignificants().values())
            
        
    def __delete_coefs_insignificants_matrix(self):
        coefs_recalculate = self.__delete_coefs_insignificants()
        for coef in self.__delete_coefs_insignificants().keys(): # scroll through dictionary keys
            if coefs_recalculate[coef] == 0: # values coef equal zero multiplies the column in matrix X  
                del self.X[coef] #save in local variable for process
        return self.X # changes atribute default
    
        
    def __executor_regression2(self):
        self.create_table_anova()
        self.plot_graphs_anova()
        self.plot_graphs_regression()
    
    def save_dataset(self):
        file = pd.ExcelWriter('dataset.xlsx')
        #coefficientes and confident interval 
        coefs_ci = pd.DataFrame({'Coef': self.model_coefients(),
                                 'coefs-ci':self.model_coefients()-self.define_ic_coefs(),
                                 'coefs+ci':self.model_coefients()+self.define_ic_coefs(),
                                 'C.I': self.define_ic_coefs(),}, index=self.X.columns)
        
        # anova
        anova = pd.DataFrame([
        [' Regression: ',self.__calculate_SSreg(),self.__df_SSreg(),self.__calculate_MSreg(),self.__ftest1() ],
        [' Residual: ',self.__calculate_SSres(), self.__df_SSres(),self.__calculate_MSres(),self.__ftest1()],
        [' Total: ', self.__calculate_SSTot(), self.__df_SSTot(),self.__calculate_MSTot(), ' F2-Test '],
        [' Pure Error: ',self.SSPE, self.df, self.__calculate_MSPE(),self.__ftest2() ],
        [' Lack of Fit: ', self.__calculate_SSLoF(), self.__df_SSLof(), self.__calculate_MSLoF(), ' F-Tabulated '],
        [' R²: ', self.__calculate_R2(), ' R: ', self.__calculate_R(), self.__ftable1() ],
        [' R² max: ',self.__calculate_R2_max(), ' R max: ', self.__calculate_R_max(),self.__ftable2()]
        ], columns=['Parameters','Sum of Squares (SS)','Degrees of Freedom (DoF)','Mean Square (MS)','F1-Test'])
        
        # predict versus experimental x coefficients
        pred_exp = pd.DataFrame(self.__matrix_X(), columns=self.X.columns)
        pred_exp['Experimental'] = self.y
        pred_exp['Predict'] = self.__calculate_pred_values()
        
        #salving data in different files
        # anova.to_excel('ANOVA.xlsx')
        # coefs_ci.to_excel('coefs_ic.xlsx')
        # pred_exp.to_excel('exp_pred.xlsx')

        #salving data in same file
        anova.to_excel(file,sheet_name= 'ANOVA')
        coefs_ci.to_excel(file,sheet_name='coefs_ic')
        pred_exp.to_excel(file,sheet_name='exp_pred')
        return file.save()
    
    
    def regression_results(self):
        
        """
        Function -> regression_results

        This function was create to calculate regression models using the equation:
        
        inv(X^tX)X^ty


        Attributes 
        -----------

        X: Matrix with the coefficients to be calculated (type: pandas.DataFrame)
        
        y: Response that will be modeled (type: pandas.Series)
        
        pqes (optional): Pure Error Sum of Squares of the values at the Central point (type: float or int)
        Use pde.CP(yc).pqes() to calculate
        For more info: help(pde.CP.pqes)
        
        dof (optional): Degrees of freedom of the central point (type: int)
        Use pde.CP(yc,k).dof_pqes()
        For more info: help(pde.CP.df_pqes)
        
        NOTE! THIS FEATURE IS STILL IN DEVELOPMENT AND DOES NOT FUNCTION WHEN THERE ARE DATA REPLICATES!
        
        auto (optional): Automate the exclusion of significant coefficients (type: bool)
        For more info: help(pde.regression.auto)
        
        self_check (optional): Automate the check for lack of model fit through analysis of variance.
        For more info: help(pde.regression.self_check)
        
        Returns
        -----------
        
        1 - ANOVA Table  (type: NoneType)
        
        2-  plot_graphs_anova() (type: NoneType) 
        
        3 - Ask about lack of fit (type: str)
        
        4- plot_graphs_regression() (type: NoneType) 
        
        
        """
        
        self.__executor_regression2()
        if self.auto == True:
            self.recalculate_coefs()
            self.__executor_regression2()
            return print(regression_analysis.__final_msg)
        else:
            return print(regression_analysis.__final_msg)
        
        
class Super_fabi:
    """
    Funcao para calcular superficie de resposta e gráfico de contorno
    A matriz X deve conter:
    Coluna 1 = coeficientes na seguinte ordem, b0, b1, b2, b11, b22, b12
    Coluna 2 = Valores codificados da variavel 1
    Coluna 3 = Valores reais da variável 1
    Coluna 4 = Valores codificados da variavel 2
    Coluna 5 = Valores reais da variavel 2
    """
    def __init__(self, coefs:list,realmax1=None,realmin1=None,realmax2=None,realmin2=None,
                 codmax1=None,codmin1=None,codmax2=None,codmin2=None):
        self.coefs = coefs
        self.realmax1 = realmax1
        self.realmin1 = realmin1
        self.realmax2 = realmax2
        self.realmin2 = realmin2
        self.codmax1 = codmax1 
        self.codmin1 = codmin1 
        self.codmax2 = codmax2 
        self.codmin2 = codmin2 
    

    def array_n1(self):
        return np.linspace(self.codmin1 ,self.codmax1 ,num=101)
    
    def array_n2(self):
        return np.linspace(self.codmin2,self.codmax2,num=101)
    
    def array_r1(self):
        return np.linspace(self.realmin1 ,self.realmax1 ,num=101)
    
    def array_r2(self):
        return np.linspace(self.realmin2,self.realmax2,num=101)
    
    def meshgrid_cod(self):
        return np.meshgrid(self.array_n1(),self.array_n2())
    
    def meshgrid_real(self):
        return np.meshgrid(self.array_r1(),self.array_r2())
    
    def z(self, meshgrid=None, x=None, y=None, manual=False):
    
        """
        Retorna valor previsto pelo modelo.
        
        Parameters 
        -----------
        
        v1: valor(es) da variável 1 (if meshgrid is True --> type numpy.array else: type float)
        
        v2: valor(es) da variável 2 (if meshgrid is True --> type numpy.array else: type float)
        
        n_var: número de variáveis a serem analisadas, por padrão n_var=2 
        
        meshgrid: (optional) (if meshgrid == True --> will be  created a matrix with Z values else: returns only a item type float)

        manual: (optional) (if manual is True --> will be calculate z value for x and y parameters ) (type:bool)
        
        """
        b0, b1, b2, b11, b22, b12 = self.coefs
        
        
        if manual == True: # if manual mode to be activate
            if x == None or y == None: # check parameters for manual mode
                raise TypeError('recalculate_coefs() está faltando 2 argumentos posicionais requirido "x" e "y".')
            else:       
                    return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        elif meshgrid == None and manual == False:
            raise TypeError('Insira parâmetros ao método.')
            
        try:
            if meshgrid == True:
                x, y = self.meshgrid_cod()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
            elif meshgrid == False:
                x, y = self.array_n1(), self.array_n2()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        except: raise TypeError('recalculate_coefs() está faltando 1 argumento posicional requirido "meshgrid".')

    def __index_max_values(self): # Return matrix meshgrid index for max value model
        idx1, idx2 = np.where(self.z(meshgrid=True) == self.z(meshgrid=True).max().max())
        return idx1[0],idx2[0]  
    
    @property
    def maxcod(self):
        """Retorna valores das coordenadas do sinal máximo para as variáveis codificadas.
        
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for codificates values  like a tuple.
        
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v2, v1 =  self.meshgrid_cod()[0], self.meshgrid_cod()[1]
        return v1[idx2][idx1], v2[idx2][idx1]
    
    @property
    def maxreal(self):
        """Retorna valores das coordenadas do sinal máximo para as variáveis codificadas.
        
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for real values like a tuple.
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v1, v2 = self.meshgrid_real()[0], self.meshgrid_real()[1]
        return v1[idx1][idx2], v2[idx1][idx2]   
 
    @property
    def zmax(self):
        r"""Retorna o valor do sinal máximo do modelo.
        
        Return
        --------
        
        fmax(x,y) = zmax like a float.
        """
        return self.z(meshgrid=True).max().max()
    
    
 
    def __etiqueta(self,matrix_X, vector_y, ax):
        vector_y = [str(j) for j in vector_y.values] # vector_y to string list 
               
        for i, label in enumerate(vector_y):
            ax.annotate(label,( matrix_X['b1'][i], matrix_X['b2'][i]),color='k',fontsize=10)  
    
    def superficie(self, matrix_X = None, vector_y = None,scatter=False):
        """Retorna os gráficos de superfície e de contorno do modelo
        
        Parameters
        ------------
        
        X: matriz X com os valores codificados dos coeficiente (type: pandas.dataframe)
        
        y: vetor y com os valores de sinais 
        
        """
        fig = plt.figure(figsize=(12,12))
        
        # Superficie de resposta
        ax1 = fig.add_subplot(1,2, 1, projection='3d')

        
        V1, V2= self.meshgrid_real()
        Z = self.z(meshgrid=True)
        b0, b1, b2, b11, b22, b12 = self.coefs
        
        surf =  ax1.plot_surface(V1, V2, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        
        ax1.set_title('Superfície de Resposta do Modelo',fontsize=12, fontweight='black',y=1,x=.55)
        ax1.set_xlabel('Variável 1')
        ax1.set_ylabel('Variável 2')
        ax1.set_zlabel('Resposta do Modelo')
        
        # Contorno 
        v1, v2= self.meshgrid_cod()
        
        ax2 = fig.add_subplot(1,2,2)
        
        contours = ax2.contour(v1, v2, Z, 3,colors='black', levels=6)
        ax2.clabel(contours, inline=True, fontsize=12)
        plt.imshow(Z, extent=[self.codmin1, self.codmax1, self.codmax2, self.codmin2],  cmap='viridis', alpha=1)
        plt.colorbar(aspect=6, pad=.15)
        
        
        if scatter == True:
            if isinstance(matrix_X,object):
                ax2.scatter(matrix_X['b1'],matrix_X['b2'], color='w',marker=(5, 1),s=50)
                self.__etiqueta(matrix_X, vector_y, ax2)
        
        ax2.scatter(self.maxcod[0], self.maxcod[1], color='darkred',marker=(5, 1),s=100)   
        ax2.annotate(r'$z_{max}= %.2f$'%self.zmax, (self.maxcod[0], self.maxcod[1]),color='k')
        
        ax2.set_xticks(list(np.arange(self.codmin1,round((2*self.codmax1),4),step=self.codmax1)) + [round(self.maxcod[0],3)])
        ax2.set_yticks(list(np.arange(self.codmin2,round((2*self.codmax2),4),step=self.codmax2)) + [round(self.maxcod[1],3)])
        ax2.set_xlabel('Variável 1')
        ax2.set_ylabel('Variável 2')
        ax2.set_title('Contorno do Modelo',fontsize=12, fontweight='black',y=1.1)
        
        fig.text(0.2,.71,r'$R_{max}(%.2f,%.2f) = %.1f\qquad\qquad\qquad  v_1^{max} = %.1f \quad e\quad v_2^{max} = %.1f $'%(self.maxcod[0],self.maxcod[1],self.zmax,self.maxreal[0],self.maxreal[1]),
                 fontsize=15,horizontalalignment='left')
        plt.suptitle(r'$\bfResposta = {} + {}v_1 + {}v_2 + {}v_1^2 + {}v_2^2 + {}v_1v_2 $'.format(b0, b1, b2, b11, b22, b12),y=.77,x=.45,fontsize=20)
        
        plt.savefig('super_fabi.png', transparent=True) 
        plt.tight_layout(w_pad=5)
        plt.show()
        

    def solver_diff(self, k=2, printf=False):
        """Método que retorna o valor máximo de resposta e os respectivos valores codificados das variáveis exploratórias do modelo através dos cálculos das derivada parciais de primeira ordem. 
        Selecione o número de variáveis através do parâmetro k. 
        Esta função é capaz de calcular para modelos com 2,3 ou 4 variáveis. 

        Parameters
        ------------

        k: número de variáveis do modelo (type: int) 

        printf (optional): Por padrão (False), será retornado valores de coordendas e resposta máxima em uma tupla e quando printf=True será retornado uma mensagem com as resposta em um display em linguagem Latex. 

        Returns
        ------------
        Retorna valores das coordenadas exploratórias para o máximo global do modelo através da derivada parcial.
        """
        v1,v2,v3,v4 = symbols('v1 v2 v3 v4', real=True) 
        
        try:
            if k == 2:           
                b0, b1, b2, b11, b22, b12 = self.coefs
                f = b0 + b1*v1 + b2*v2 + b11*v1**2 + b22*v2**2 + b12*v1*v2
                X = np.matrix([
                     [2*b11,b12],
                     [b12,2*b22]])
                y = -np.array(self.coefs[1:3])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1])])

                resultados = np.append(p_diff,fmax)
                
                if printf == False:
                    return pd.DataFrame({'Resultados':resultados},index=['b1','b2','Resposta'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f})= {2:.2f}$$".format(p_diff[0],p_diff[1],fmax)))

            elif k == 3:   
                b0, b1, b2, b3, b11, b22, b33, b12, b13, b23 = self.coefs
                f = b0 + b1*v1 + b2*v2 + b3*v3 + b11*v1**2 + b22*v2**2 + b33*v3**2 + b12*v1*v2 + b13*v1*v3 + b23*v2*v3
                X = np.matrix([
                     [2*b11,b12,b13],
                     [b12,2*b22,b23],
                     [b13,b23,2*b33]])
                y = -np.array(self.coefs[1:4])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame({'Resultados':resultados},index=['b1','b2','b3','Resposta'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f},{2:.3f})= {3:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],fmax)))

            elif k == 4:  
                b0, b1, b2, b3, b4, b11, b22, b33, b44, b12, b13, b14, b23, b24, b34 = self.coefs
                f=b0+b1*v1+b2*v2+b3*v3+b4*v4+b11*v1**2+b22*v2**2+b33*v3**2+b44*v4**2+b12*v1*v2+b13*v1*v3+b14*v1*v4+b23*v2*v3+b24*v2*v4+b34*v3*v4
                X = np.matrix([
                     [2*b11,b12,b13,b14],
                     [b12,2*b22,b23,b24],
                     [b13,b23,2*b33,b34],
                     [b14,b24,b34,2*b44]])
                y = -np.array(self.coefs[1:5])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2]),(v4,p_diff[3])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame( {'Resultados':resultados},index=['b1','b2','b3','b4','Resposta'])
                else:
                    return display(Latex("$$f'({0:.4f},{1:.3f},{2:.3f},{3:.3f})= {4:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],
                                                                                                   p_diff[3],fmax)))
        except: raise TypeError(f'Não há registro para solução da equação para "k == {k}"')