import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns             #library for analysis

from scipy.stats import t         #library for regression

from scipy.stats import f
from scipy.stats import linregress
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import iplot
from IPython.display import display, Latex, Markdown
import sys                        #library for surface graph


from sympy import symbols         #sympy library for symbols, diff, solve and subs

class Analysis: 
    
    """
    Class -> Analysis(X,y,effect_error,t) - Class for calculating factorial planning effects.
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

        # Rotate x-axis labels for the bar plot
        for tick in axs[1].get_xticklabels():
            tick.set_rotation(45)
        
        fig.suptitle('Effect Analysis Chart', fontsize=22, y=0.99, fontweight='black', color='darkred')
        plt.tight_layout()
        plt.savefig('effect_analysis_Chart.png', transparent=True)
        
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

    def model_equation(self):

        """
        Function -> model_equation
        This function is designed to print and save the effect coefficient, the b0 coefficient, and the first model equation.
        """
        #inputs
        avg_y = self.y.mean()
        effects = self.__calculate_effects()
        coefficients = effects / 2
    
        # Construicting the DataFrame
        data = pd.DataFrame({
            'Coefficients': coefficients,
            # 'Percentage': self.__calculate_percentage_effects()
        }, index=self.__effect_indices)
    
        # Constructing the equation string
        equation = f"R = {avg_y.round(5)} + {coefficients[0].round(5)}*v1 + {coefficients[1].round(5)}*v2 + {coefficients[2].round(5)}*v1v2"
        
        # output
        display(data)
        display(Markdown(f'**The intercept (b0) is equal to:** {avg_y}'))
        display(Markdown('**Linear Equation**'))
    
        print(equation)

    def plot_surface(self):
        """
        Function -> plot_surface
        This function is designed to plot and save the surface graph for the first approximation.
        """
        # inputs
        effects = self.__calculate_effects()
        avg_y = self.y.mean()
        coefficients = effects / 2
    
        # Create grid and compute Z values
        v1 = np.linspace(-1, 1, 100)
        v2 = np.linspace(-1, 1, 100)
        v1, v2 = np.meshgrid(v1, v2)
        R = avg_y + coefficients[0] * v1 + coefficients[1] * v2 + coefficients[2] * v1 * v2
    
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(v1, v2, R, cmap='viridis')
        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_zlabel('R')
        ax.set_title('Surface Plot of the Linear Equation')
        equation_str = r'$R = {:.5f} {:+.5f}v_1 {:+.5f}v_2 {:+.5f}v_1v_2$'.format(avg_y, coefficients[0], coefficients[1], coefficients[2])
        plt.suptitle(equation_str, y=1, x=0.45, fontsize=15)
        
        # Constructing the equation string
        equation = "Linear Equation: R = {:.5f} {:+.5f}*v1 {:+.5f}*v2 {:+.5f}*v1*v2".format(avg_y, coefficients[0], coefficients[1], coefficients[2])

        #output
        print(equation)
        print('---------------------------------------------------------')
        plt.show()
        plt.savefig('Surface plot-Linear equation.png',transparent=True)

    def plot_surface3D(self):
        """
        Function -> plot_surface
        This function is designed to plot a 3D the surface graph for the first approximation.
        """
        # input
        effects = self.__calculate_effects()
        avg_y = self.y.mean()
        coefficients = effects / 2

        # Create grid and compute Z values
        v1 = np.linspace(-1, 1, 100)
        v2 = np.linspace(-1, 1, 100)
        v1, v2 = np.meshgrid(v1, v2)
        R = avg_y + coefficients[0] * v1 + coefficients[1] * v2 + coefficients[2] * v1 * v2
    
        # Create the surface plot
        surface = go.Surface(z=R, x=v1, y=v2, colorscale='Viridis')
        layout = go.Layout(
            title='3D Surface Plot of the Linear Equation',
            scene=dict(
                xaxis=dict(title='v1'),
                yaxis=dict(title='v2'),
                zaxis=dict(title='R')
            ),

            width=800,  # Adjust width as needed
            height=600,  # Adjust height as needed
        )

        # Constructing the equation string
        equation = "Linear Equation: R = {:.5f} {:+.5f}*v1 {:+.5f}*v2 {:+.5f}*v1*v2".format(avg_y, coefficients[0], coefficients[1], coefficients[2])
  
        #output
        fig = go.Figure(data=[surface], layout=layout)
        print(equation)
        print('---------------------------------------------------------')
        iplot(fig)

class ExpStat:
    """
    ExpStat(yr, n, k) - A class for calculating experimental statistics.

    Attributes
    ----------
    yr : pd.Series
        Values of experimental replicates.
    n : int
        Number of replicates.
    k : int
        Number of variables

    Method
    ----------
    results(): print (Experimental Variance, Experimental Error, Effect Error and t-Student)
    """
    def __init__(self,yr=None, n=None, k =None):
        self.yr = yr
        self.n = n
        self.k = k

    def __mean_yr__(self):
        # calculating mean of replicates
        if self.yr is None:
            raise ValueError("yr attribute must be specified.")
        
        mean_yr = self.yr.mean(axis=1)
        return mean_yr

    def __var_yr__(self):
        # calculating variance
        if self.yr is None:
            raise ValueError("yr attribute must be specified.")
        
        var_yr = self.yr.var(axis=1)
        return var_yr
    def __invt__(self):
        # calculating t-Student
        numb_row = self.yr.shape[0] #number of rows
        dof = (numb_row*(self.n-1)) # degree of freedom
        
        return t.ppf(1-.05/2,dof)

    # def __calculate_sspe(self):
    
    #     mean_yr = np.mean(self.yr)  # Calculate mean of yr
    #     sspe = np.sum((self.yr - mean_yr)**2)  # Calculate SSPE

    #     return sspe 

    def results(self):
        exp_var = self.__var_yr__().mean()                       # experimental variance = avarege variance
        exp_error = exp_var**0.5                                 # experimental error = sqrt(experimental variance)
        effect_error = ((2*exp_error)/(self.n*2**self.k)**0.5)   # effect error = ((2*exp_error)/(n*2**k)**0.5)
        t = self.__invt__()                                      # t-Student
        # sspe = self.__calculate_sspe()                           # square sum of pure error

        display(Markdown(f'**Experimental Variance:** {exp_var}'))
        display(Markdown(f'**Experimental Error:** {exp_error}'))    
        display(Markdown(f'**Effect Error:** {effect_error}'))
        display(Markdown(f'**t-Student:** {t}'))
        # display(Markdown(f'**SSPE:** {sspe}'))
      

        
class Cp:
    """
    Class -> Cp(y, k) - Class for calculating value-t and effect_error 
        
    Atributes
    -----------
    y: pd.Series - values of center points region
    
    k: int -  number of variables 
    
    Methods
    -----------
    invt: return t-value.
    
    effect_error: return the error of one efffect
    
    sspe: return Pure Quadratic Error Sum
    
    df_sspe: return the degrees of freedom
   
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
        
        (optional) df_a: degree of freedom that not belong to Cp class
        
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
    
    def __calculate_sspe(self):
     
        return np.sum((self.__array() - np.mean(self.__array()))**2)
    
    def sspe(self):     
        """Return sspe value"""
        if self.y.all() == None:
            return self.__message_error_11()
        else:
            sspe_value = self.__calculate_sspe()
            display(Markdown(f'**SSPE is equal to** {sspe_value}'))
            return sspe_value
             
            
    def  df_sspe(self):
        """Return degree of freedom"""
        return len(self.y)

class RegressionAnalysis:
    """
        Class -> RegressionAnalysis(X, y, sspe, dof) - Create a regression model and adjust it through Variance Analysis
        
        This routine aims to calculate regression models using the following equation:
        
        $inv(X^tX)X^ty$
        
        Attributes
        -------------------
        X: Matrix with the coefficients to be calculated (type: pandas.DataFrame)
        
        y: Response that will be modeled (type: pandas.Series)
        
        sspe (optional): Pure Error Sum of Squares of the values at the Central point (type: float or int)
        Use doe.Cp(yc).sspe() to calculate
        For more info: help(doe.Cp.sspe)
        
        dof (optional): Degrees of freedom of the central point (type: int)
        Use doe.Cp(yc,k).dof_sspe()
        For more info: help(doe.CP.df_sspe)
        
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
        
        recalculate_coeffs: Returns a pandas.DataFrame with the significant model coefficients
        
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
    
    def __calculate_var_coeffs(self):
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
    
    def calculate_coeffs(self):
        """Return the sum of restult of the definition: "__matrix_coef" """
        return np.einsum('ij->j', self.__calculate_matrix_coef()).round(5)
    
    
    def __calculate_pred_values(self):
        """Retunr the values of predict by the model"""
        return np.matmul(self.X,self.calculate_coeffs())
    
 
    def predict(self, value=0):
        """Retunr the values of predict by the model"""
        return np.matmul(self.X,self.calculate_coeffs()+value)
    
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

        axs0[0,1].bar('MSRes and t',self.__calculate_MSres(),color='darkorange')
        axs0[0,1].set_title('MQ Residual',fontweight='black')
        axs0[0,1].text(-.35,.5, '%.1f  %.3f'%(self.__calculate_MSres(),Cp().invt(self.__df_SSres()-1)), fontsize=20,color='k')
        #axs0[0,1].text(-.35, 2.07, '%.4f'%Cp().invt(self.__df_SSres()-1), fontsize=20,color='k')

        axs0[1,0].bar('MSPE',3, color= 'darkred')
        axs0[1,0].set_title('MS of Pure Error',fontweight='black')
        axs0[1,0].text(-.35, 1.27,'%.2f'%self.__calculate_MSPE(), fontsize=20,color='w')

        axs0[1,1].bar('MSLoF and t',3,color= 'darkviolet')
        axs0[1,1].set_title('MS of Lack of Fit',fontweight='black')
        axs0[1,1].text(-.35, 1.98, '%.1f'%self.__calculate_MSLoF(), fontsize=20,color='w')
        axs0[1,1].text(-.35, 1.07, '%.4f'%Cp().invt(self.__df_SSLof()), fontsize=20,color='w')

        
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
        
     
        fig.suptitle('ANOVA (Analisys of Variance)', fontsize=18, fontweight='black',y=1.05)
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
    
    
    def define_ic_coeffs(self,msg=False): #decides if will calculate ic mslof or msres
        if self.self_check == False:
            check_answer = self.__check_model() #Returns True or False through this method
        elif self.self_check == True:
            check_answer = self.__self_turning(msg)
        else:
            raise TypeError('"doe.RegressionAnalysis().resgression_results()" one of argument is missing for "self.check".')
           
        if check_answer == True:
            RegressionAnalysis.__check_ci = True # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSLoF() # to calculate interval confidence for lack of fit
        elif check_answer == False:
            RegressionAnalysis.__check_ci = False # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSRes() #to calculate interval confidence for residues 
        
    def show_ci(self, manual=None):
        """Absolute values of model's confident interval"""
        if RegressionAnalysis.__check_ci == True or manual ==True:
            return self.__define_ic_MSLoF()
        elif  RegressionAnalysis.__check_ci == False or manual == False:
            return self.__define_ic_MSRes()
    
    def __define_ic_MSLoF(self): #calculates confidence interval for mslof
        return (((self.__calculate_MSLoF()*self.__calculate_var_coeffs())**0.5)*Cp().invt(self.__df_SSLof()-1)).round(4)
        
    def __define_ic_MSRes(self): #calculates confidence interval for msresN
        return (((self.__calculate_MSres()*self.__calculate_var_coeffs())**0.5)*Cp().invt(self.__df_SSres()-1)).round(4)
    
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
        # fig.subplots_adjust(top=0.95, hspace=0.5, wspace=0.4)
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
        
        axs3.errorbar(self.X.columns,self.calculate_coeffs(),self.define_ic_coeffs(True), fmt='^', linewidth=2, capsize=6, color='darkred')
        axs3.axhline(0,color='darkred', linestyle='dashed')
        axs3.set_ylabel('Coefficient Values')
        axs3.set_xlabel('Coefficient')
        axs3.set_title('Regression of Coefficients',fontweight='black')
        axs3.grid()
        
        fig.suptitle('Regression Model', fontsize=18, fontweight='black',y=0.95)
        plt.savefig('Regression Model.png',transparent=True)
        
        return plt.show()
    
    
    # Recalculate the model and to variables insignificant excludes automatically    
    
    def dict_coeffs_ci(self): #list with dicts {'coeffs':values,'coeffs_max':values,'coeffs_min':values}
        return  [dict(zip(self.X.columns, self.calculate_coeffs())),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coeffs()[0].round(4))),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coeffs()[1].round(4)))]
    
    def recalculate_coeffs(self):  #returns an array with coeffs values and coeffs insignificants equal zero 
        """Return a DataFrame with significants coefficients"""
        return self.__delete_coeffs_insignificants_matrix()
       
    def __calculate_inter_max_min_coeffs(self): #returns a tuple with (coef+ci,coef-ci)
        if RegressionAnalysis.__check_ci == True:
            return [self.calculate_coeffs() + self.__define_ic_MSLoF(), 
                            self.calculate_coeffs() - self.__define_ic_MSLoF()]
        elif  RegressionAnalysis.__check_ci == False:
            return [self.calculate_coeffs() + self.__define_ic_MSRes(),
                            self.calculate_coeffs() - self.__define_ic_MSRes()] 
    
        
    def __delete_coeffs_insignificants(self): #select (coef - ci <= coef <= coef + ci) and replace for zero
        coeffs = self.dict_coeffs_ci()[0]
        max_ = self.dict_coeffs_ci()[1]
        min_ = self.dict_coeffs_ci()[2]
        for coef in coeffs.keys():
            if min_[coef]<= 0 <= max_[coef]:
                coeffs[coef] = 0
        return coeffs
    
    def model_coeffients(self):
        """Return a values list with significants coefficients and null values for insignificants coefficients"""
        return list(self.__delete_coeffs_insignificants().values())
            
        
    def __delete_coeffs_insignificants_matrix(self):
        coeffs_recalculate = self.__delete_coeffs_insignificants()
        for coef in self.__delete_coeffs_insignificants().keys(): # scroll through dictionary keys
            if coeffs_recalculate[coef] == 0: # values coef equal zero multiplies the column in matrix X  
                del self.X[coef] #save in local variable for process
        return self.X # changes atribute default
    
        
    def __executor_regression(self):
        self.create_table_anova()
        self.plot_graphs_anova()
        self.plot_graphs_regression()
    
    def save_dataset(self):
        file_path = 'dataset.xlsx'
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        
            #coefficientes and confident interval 
            coeffs_ci = pd.DataFrame({
                'Coef': self.model_coeffients(),
                'coeffs-ci':self.model_coeffients()-self.define_ic_coeffs(),
                'coeffs+ci':self.model_coeffients()+self.define_ic_coeffs(),
                'C.I': self.define_ic_coeffs(),}, index=self.X.columns)
            coeffs_ci.to_excel(writer, sheet_name='coeffs_ic')
            
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
            anova.to_excel(writer, sheet_name='ANOVA')
            
            # predict versus experimental x coefficients
            pred_exp = pd.DataFrame(self.__matrix_X(), columns=self.X.columns)
            pred_exp['Experimental'] = self.y
            pred_exp['Predict'] = self.__calculate_pred_values()
            pred_exp.to_excel(writer, sheet_name='exp_pred')
                
        print("dataset.xlsx was saved on your directory")
        return file_path
    
    
    def regression_results(self):
        
        """
        Function -> regression_results

        This function was create to calculate regression models using the equation:
        
        inv(X^tX)X^ty


        Attributes 
        -----------

        X: Matrix with the coefficients to be calculated (type: pandas.DataFrame)
        
        y: Response that will be modeled (type: pandas.Series)
        
        sspe (optional): Pure Error Sum of Squares of the values at the Central point (type: float or int)
        Use doe.Cp(yc).sspe() to calculate
        For more info: help(doe.Cp.sspe)
        
        dof (optional): Degrees of freedom of the central point (type: int)
        Use pde.Cp(yc,k).dof_sspe()
        For more info: help(doe.Cp.df_sspe)
        
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
        
        self.__executor_regression()
        if self.auto == True:
            self.recalculate_coeffs()
            self.__executor_regression()
            return print(RegressionAnalysis.__final_msg)
        else:
            return print(RegressionAnalysis.__final_msg)
        
        
class Surface: 
    """
    Class --> Surface (coeffs, realmax1, realmin1, realmax2, realmin2, codmax1, codmin1, codmax2, codmin2 ) to calculating the surface graph and contour chart
    TThe matrix must contain the following columns:
    Column 1 = coefficients in the following sequence: b0, b1, b2, b11, b22, b12
    Column 2 = encoded values for variable 1
    Column 3 = real values for variable 1
    Column 4 = encoded values for variable 2
    Column 5 = real values for variable 2

    """
    def __init__(self, coeffs:list,realmax1=None,realmin1=None,realmax2=None,realmin2=None,
                 codmax1=None,codmin1=None,codmax2=None,codmin2=None):
        self.coeffs = coeffs
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
        Return predict values from model.
        
        Parameters 
        -----------
        
        v1: values of variable 1 (if meshgrid is True --> type numpy.array else: type float)
        
        v2: values of variable 2 (if meshgrid is True --> type numpy.array else: type float)
        
        n_var: number of variables to be analysed. Default n_var=2 
        
        meshgrid: (optional) (if meshgrid == True --> will be  created a matrix with Z values else: returns only a item type float)

        manual: (optional) (if manual is True --> will be calculate z value for x and y parameters ) (type:bool)
        
        """
        b0, b1, b2, b11, b22, b12 = self.coeffs
        
        
        if manual == True: # if manual mode to be activate
            if x == None or y == None: # check parameters for manual mode
                raise TypeError('recalculate_coeffs() 2 required positional arguments "x" and "y" are missing.')
            else:       
                    return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        elif meshgrid == None and manual == False:
            raise TypeError('Insert parameters to the method.')
            
        try:
            if meshgrid == True:
                x, y = self.meshgrid_cod()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
            elif meshgrid == False:
                x, y = self.array_n1(), self.array_n2()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        except: raise TypeError('recalculate_coeffs() 1 required positional arguments "meshgrid" is missing.')

    def __index_max_values(self): # Return matrix meshgrid index for max value model
        idx1, idx2 = np.where(self.z(meshgrid=True) == self.z(meshgrid=True).max().max())
        return idx1[0],idx2[0]  
    
    @property
    def maxcod(self):
        """
        Returns values ​​of the maximum signal coordinates for the encoded variables.
                
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for codificates values like a tuple.
        
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v2, v1 =  self.meshgrid_cod()[0], self.meshgrid_cod()[1]
        return v1[idx2][idx1], v2[idx2][idx1]
    
    @property
    def maxreal(self):
        """Returns values ​​of the maximum signal coordinates for the encoded variables.
        
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for real values like a tuple.
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v1, v2 = self.meshgrid_real()[0], self.meshgrid_real()[1]
        return v1[idx1][idx2], v2[idx1][idx2]   
 
    @property
    def zmax(self):
        """Returns the maximum signal value of the model.
        
        Return
        --------
        
        fmax(x,y) = zmax like a float.
        """
        return self.z(meshgrid=True).max().max()
    
    
 
    def __tag(self,matrix_X, vector_y, ax):
        vector_y = [str(j) for j in vector_y.values] # vector_y to string list 
               
        for i, label in enumerate(vector_y):
            ax.annotate(label,( matrix_X['b1'][i], matrix_X['b2'][i]),color='k',fontsize=10)  
    
    def surface_results(self, matrix_X = None, vector_y = None,scatter=False):
        """Return surface graph and countour chart
        
        Parameters
        ------------
        
        X: matrix X with encoded values of coefficient (type: pandas.dataframe)
        
        y: vector y with signal values
        
        """
        fig = plt.figure(figsize=(12,12))
        
        # Superface result
        ax1 = fig.add_subplot(1,2, 1, projection='3d')

        
        V1, V2= self.meshgrid_real()
        Z = self.z(meshgrid=True)
        b0, b1, b2, b11, b22, b12 = self.coeffs
        
        surf =  ax1.plot_surface(V1, V2, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        
        ax1.set_title('Model Surface',fontsize=12, fontweight='black',y=1,x=.55)
        ax1.set_xlabel('Variable 1')
        ax1.set_ylabel('Variable 2')
        ax1.set_zlabel('Model Predict')
        
        # Contour
        v1, v2= self.meshgrid_cod()
        
        ax2 = fig.add_subplot(1,2,2)
        
        contours = ax2.contour(v1, v2, Z, 3,colors='black', levels=6)
        # contours = ax2.contour(V1, V2, Z, 3,colors='black', levels=6) #change for real values, maybe
        ax2.clabel(contours, inline=True, fontsize=12)
        plt.imshow(Z, extent=[self.codmin1, self.codmax1, self.codmax2, self.codmin2],  cmap='viridis', alpha=1)
        # plt.imshow(Z, extent=[self.realmin1, self.realmax1, self.realmax2, self.realmin2],  cmap='viridis', alpha=1)
        plt.colorbar(aspect=6, pad=.15)
        
        
        if scatter == True:
            if isinstance(matrix_X,object):
                ax2.scatter(matrix_X['b1'],matrix_X['b2'], color='w',marker=(5, 1),s=50)
                self.__tag(matrix_X, vector_y, ax2)
        
        ax2.scatter(self.maxcod[0], self.maxcod[1], color='darkred',marker=(5, 1),s=100)   
        ax2.annotate(r'$z_{max}= %.2f$'%self.zmax, (self.maxcod[0], self.maxcod[1]),color='k')
        
        ax2.set_xticks(list(np.arange(self.codmin1,round((2*self.codmax1),4),step=self.codmax1)) + [round(self.maxcod[0],3)])
        ax2.set_yticks(list(np.arange(self.codmin2,round((2*self.codmax2),4),step=self.codmax2)) + [round(self.maxcod[1],3)])
        ax2.set_xlabel('Variable 1')
        ax2.set_ylabel('Variable 2')
        ax2.set_title('Contour Chart of Model',fontsize=12, fontweight='black',y=1.1)

        #print and save
        fig.text(0.2, 0.71, r'$R_{max}(%.2f, %.2f) = %.1f\qquad\qquad\qquad  v_1^{max} = %.1f \quad e\quad v_2^{max} = %.1f$' % (self.maxcod[0], self.maxcod[1], self.zmax, self.maxreal[0], self.maxreal[1]),
        fontsize=15, horizontalalignment='left')

        plt.suptitle(r'$f(v) = {} + {}v_1 + {}v_2 + {}v_1^2 + {}v_2^2 + {}v_1v_2$'.format(b0, b1, b2, b11, b22, b12),y=0.77, x=0.45, fontsize=15)

        plt.tight_layout(w_pad=5)
        plt.savefig('surface.png', transparent=True) 
        plt.show()

    #def #adicionar a funcionalidade de interagir e rotacionar o gráfico de superfície utilizando o plotly.graph_objs as go
        # def surface_rotate (self, matrix_X=None, vector_y=None, scatter=False):
        #     # Assuming self.meshgrid_real() and self.z(meshgrid=True) return necessary data
        #     V1, V2 = self.meshgrid_real()
        #     Z = self.z(meshgrid=True)
        #     b0, b1, b2, b11, b22, b12 = self.coeffs
    
        #     # Criar figura de superfície no Plotly
        #     surface_plot = go.Figure(data=[go.Surface(x=V1, y=V2, z=Z, colorscale='Viridis')])
    
        #     # Configurar layout da superfície
        #     surface_plot.update_layout(
        #         title='Model Surface',
        #         scene=dict(
        #             xaxis_title='Variable 1',
        #             yaxis_title='Variable 2',
        #             zaxis_title='Model Predict',
        #             camera=dict(
        #                 up=dict(x=0, y=0, z=1),
        #                 center=dict(x=0, y=0, z=0),
        #                 eye=dict(x=1.25, y=1.25, z=1.25)
        #             ),
        #             aspectratio=dict(x=1, y=1, z=0.7),
        #             aspectmode='manual'
        #         )
        #     )

        #     # Adicionar gráfico de dispersão se scatter=True
        #     if scatter:
        #         # Adicionar um gráfico de dispersão 3D
        #         scatter_data = go.Scatter3d(
        #             x=matrix_X.iloc[:, 0],
        #             y=matrix_X.iloc[:, 1],
        #             z=vector_y,
        #             mode='markers',
        #             marker=dict(size=5, color='red')
        #         )
        #         surface_plot.add_trace(scatter_data)
            
        #     # Mostrar o gráfico de superfície
        #     surface_plot.show()
        

    def solver_diff(self, k=2, printf=False):
        """Method that returns the maximum response value and the respective coded values ​​of the model's exploratory variables through calculations of first-order partial derivatives. 
        Select the number of variables using the k parameter. 
        This function is capable of calculating for models with 2,3 or 4 variables. 

        Parameters
        ------------

        k: number of model variables (type: int) 

        printf (optional): By default (False), it will return coordinate values and the maximum response in a tuple. When printf=True, it will return a message with the responses displayed in Latex language.

        Returns
        ------------
        Returns values ​​of the exploratory coordinates for the global maximum of the model through the partial derivative.
        """
        v1,v2,v3,v4 = symbols('v1 v2 v3 v4', real=True) 
        
        try:
            if k == 2:           
                b0, b1, b2, b11, b22, b12 = self.coeffs
                f = b0 + b1*v1 + b2*v2 + b11*v1**2 + b22*v2**2 + b12*v1*v2
                X = np.matrix([
                     [2*b11,b12],
                     [b12,2*b22]])
                y = -np.array(self.coeffs[1:3])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y matrix and vector multiplication (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1])])

                results = np.append(p_diff,fmax)
                
                if printf == False:
                    return pd.DataFrame({'Results':results},index=['b1','b2','Awnser'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f})= {2:.2f}$$".format(p_diff[0],p_diff[1],fmax)))

            elif k == 3:   
                b0, b1, b2, b3, b11, b22, b33, b12, b13, b23 = self.coeffs
                f = b0 + b1*v1 + b2*v2 + b3*v3 + b11*v1**2 + b22*v2**2 + b33*v3**2 + b12*v1*v2 + b13*v1*v3 + b23*v2*v3
                X = np.matrix([
                     [2*b11,b12,b13],
                     [b12,2*b22,b23],
                     [b13,b23,2*b33]])
                y = -np.array(self.coeffs[1:4])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame({'Results':results},index=['b1','b2','b3','Answer'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f},{2:.3f})= {3:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],fmax)))

            elif k == 4:  
                b0, b1, b2, b3, b4, b11, b22, b33, b44, b12, b13, b14, b23, b24, b34 = self.coeffs
                f=b0+b1*v1+b2*v2+b3*v3+b4*v4+b11*v1**2+b22*v2**2+b33*v3**2+b44*v4**2+b12*v1*v2+b13*v1*v3+b14*v1*v4+b23*v2*v3+b24*v2*v4+b34*v3*v4
                X = np.matrix([
                     [2*b11,b12,b13,b14],
                     [b12,2*b22,b23,b24],
                     [b13,b23,2*b33,b34],
                     [b14,b24,b34,2*b44]])
                y = -np.array(self.coeffs[1:5])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2]),(v4,p_diff[3])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame( {'Results':results},index=['b1','b2','b3','b4','Answer'])
                else:
                    return display(Latex("$$f'({0:.4f},{1:.3f},{2:.3f},{3:.3f})= {4:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],
                                                                                                   p_diff[3],fmax)))
        except: raise TypeError(f'There is no record for solving the equation for "k == {k}"')

class DataPrint:
    """
    Class to print the variables set
    DataPrint(X,y,yc,effect_error,t)
    If you doon't have the central points
    
    """

    def __init__(self, **kwargs):
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)
        self.yc = kwargs.get('yc', None)
        # self.effect_error = kwargs.get('effect_error', None)
        # self.t = kwargs.get('t', None)

    def variables(self):
        if self.X is not None:
            display(Markdown('**Matrix X**'))
            print(self.X)
        else:
            display(Markdown('**Check if Matrix X is set**'))
        print()

        if self.y is not None:
            display(Markdown('**Vector y**'))
            print(self.y)
        else:
            display(Markdown('**Check if Vector y is set**'))
        print()

        if self.yc is not None:
            display(Markdown('**The Central Points are:**'))
            print(self.yc)
        else:
            display(Markdown('**Check if yc replica is set**'))
        print()

    # def statistical_values
