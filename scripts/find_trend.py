import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from util import clean_data
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression


def calculate_corr_from_total_syanapses(path):
    
    #read data
    df = pd.read_excel(path)

    #Make column names for dataframe
    labels = df.columns.to_list()
    calculations = ['Pearsons correlation', 
                    'Pearsons p-value', 
                    'Spearmans correlation', 
                    'Spearmans p-value']
    labels.extend(calculations)
    
    #Give timepoints for calculating correlation
    timepoints = [5, 8, 16, 23, 27, 50, 50]
    #timepoints= [0, 5, 8, 16, 23, 27, 50, 50]
    
    #make empty list to hold all final values
    with_cor = []

    #convert data into lists
    lists = df.values.tolist()

    #Iterate over values to calculate correlation and pvalues
    for list in lists:
        values = list[1:-1]

        pearsons = sc.stats.pearsonr(timepoints, values)[0]
        pearsons_pvalue = sc.stats.pearsonr(timepoints, values)[1]

        spearmans = sc.stats.spearmanr(timepoints, values).correlation
        spearmans_pvalue = sc.stats.spearmanr(timepoints, values).pvalue

        list.extend([pearsons, pearsons_pvalue, spearmans, spearmans_pvalue])
        
        with_cor.append(list)
    
    #Generate dataframe with calculated correlation values and their corresponding p-values
    df_wcor = pd.DataFrame(with_cor, columns=labels)

    return df_wcor

def filter_by_pvalues(data, cutoff):

    split_by = data['Pearsons p-value'] < cutoff

    #filter for connections with pearson's pvalue < cutoff
    pearson = data[split_by]
    remainder = data[~split_by]

    #find the number of connections with only Spearman pvalues < the cutoff 
    spearman_mask = remainder['Spearmans p-value'] < cutoff
    spearman_only = remainder[spearman_mask]

    no_trend = remainder[~spearman_mask]

    return pearson, spearman_only, no_trend

def get_ci(data, alpha = 0.95):
    
    lp = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(data, lp)
    
    up = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(data, up)

    return lower, upper

def find_neural_age (data, name, nonparametric = True):
    
    data.set_index('Neurons', inplace = True)
    #change this as needed

    #filter out the correlation calculations
    all = data.iloc[:, :8].T
    #leave out the testing point
    ax = data.iloc[:, :7].T

    ax = ax.rename(index = {'L1_3': 5 ,'L1_4': 8, 'L1_5':16, 'L2': 23, 'L3':27, 'TEM adult':50, 'SEM adult': 50})
    
    timepoints =np.array([[5, 8, 16, 23, 27, 50, 50]])
    xmin = min(timepoints[0])
    xmax = max(timepoints[0])

    mapping = {}
    all_neural_ages = []
    ci = [] 

    for neuron in ax.columns:
        #if neuron == 'AVA':
        x, y = ax[neuron].index.values, ax[neuron].values
        con_name = neuron
        pp = PdfPages(Path(f'./graphs/nonparametric_bootstrapping_{name}_{con_name}.pdf'))
        fig = plt.figure(figsize=(16,6))
        ax0 = fig.add_subplot(121)

        #plot data as scatter
        ax0.scatter(x,y, marker='o', color='#fc8d62', zorder=4)

        #plot xlabels and title
        ax0.title.set_text(neuron)
        ax0.set_xticks(timepoints[0][:6])
        ax0.set_xticklabels(['L1_3 (5)','L1_4 (8)', 'L1_5 (16)', 'L2 (23)','L3 (27)', 'Adult (50)'])

        # Extend x data to contain another row vector of 1s
        X = np.vstack([x,np.ones(len(x))]).T

        #Perform linear regression
        lr = LinearRegression()
        lr.fit(X, y)

        #Find the regression values at each timepoint as well as the residuals
        y_predict = lr.predict(X)
        residuals = y- y_predict

        #Get test point value
        test_point= all[neuron].values[-1]

        #Plot test point (horizontal line) and the regression line
        ax0.hlines(test_point,xmin, xmax, linestyles='dashed', color='#d7191c', zorder = 5)
        ax0.plot(x, lr.predict(X), color = '#0571b0', zorder=2)

        #Get slope and y-intercept for regression line
        slope = lr.coef_[0]
        y_intercept = lr.intercept_

        
        #Using the above slope and intercept to caculate x value (neural age)
        testx = (test_point - y_intercept)/slope 

        # Add neural age cacluated based on real data to a dictionary, and slope
        mapping[neuron] = [testx, slope]

        #making lists for slopes, intercepts and neural ages for bootstrapping
        bs_slopes = []
        bs_intercepts = []
        bs_neural_ages = []
        
        #draw bootstrapping lines
        n_boots = 1000
        for i in range(n_boots):
            if nonparametric == True:
                # create a sampling of the residuals with replacement
                boot_resids = np.random.choice(residuals, len(y), replace=True)
                y_temp = [y_predict + residuals for y_predict, residuals in zip(y_predict, boot_resids)]

                lr = LinearRegression()
                line = lr.fit(X, y_temp)            
            
            else:
                #randomly sample from data
                sample_index = np.random.choice(range(0, len(y)), len(y), replace = True)

                x_sample = X[sample_index]
                y_sample = y[sample_index]

                lr = LinearRegression()
                line = lr.fit(x_sample, y_sample)

            m = line.coef_[0]
            b = line.intercept_

            if m == 0:
                neural_age = b

            else:
                neural_age = (test_point - b)/m 
            
            #Add the slope and intercept to the master list
            bs_slopes.append(m)
            bs_intercepts.append(b)
            bs_neural_ages.append(neural_age)
            
            y_vals = m * x + b
            plt.plot(x, y_vals, color='grey', alpha=0.2, zorder=1)
            if max(y) > 20:
                ax0.set_yticks([0,5,10,15,20,25,30,35])

        
        transformed_nage = clean_data(bs_neural_ages, [0, 50])
        data1 = fig.add_subplot(122)
        data1.hist(transformed_nage, 20)
        data1.title.set_text(f'{neuron} Neural Age')
        
        pp.savefig(fig)
        plt.close()
        
        all_neural_ages = all_neural_ages + bs_neural_ages

        #Find 95% confidence interval of the neural ages
        ci_lo_nage, ci_hi_nage = get_ci(bs_neural_ages)

        # add confidence interval to master list
        ci.append([ci_lo_nage, ci_hi_nage])
        
        pp.close()

def inside_outside(data, name):

    data.set_index('Neurons', inplace = True)
    #change this as needed

    #filter out the correlation calculations
    all = data.iloc[:, :8].T
    #leave out the testing point
    ax = data.iloc[:, :7].T

    ax = ax.rename(index = {'L1_3': 5 ,'L1_4': 8, 'L1_5':16, 'L2': 23, 'L3':27, 'TEM adult':50, 'SEM adult': 50})
    
    timepoints =np.array([[5, 8, 16, 23, 27, 50, 50]])
    xmin = min(timepoints[0])
    xmax = max(timepoints[0])

    for neuron in ax.columns:
        #if neuron == 'AVA':
        x, y = ax[neuron].index.values, ax[neuron].values
        con_name = neuron
        pp = PdfPages(Path(f'./graphs/inside_outside_{name}_{con_name}.pdf'))
        fig = plt.figure(figsize=(16,6))

        #plot data as scatter
        plt.scatter(x,y, marker='o', color='#fc8d62', zorder=4)

        #plot xlabels and title
        labels = ['L1_3 (5)','L1_4 (8)', 'L1_5 (16)', 'L2 (23)','L3 (27)', 'Adult (50)']
        plt.title(neuron)
        plt.xticks(ticks = timepoints[0][:6], labels = labels)

        #calculate standard deviation
        avg = np.mean(y)
        std = np.std(y)
        plt.fill_between(x, avg-2*std, avg+2*std, color = '#9ecae1')
        
        #Get test point value
        test_point= all[neuron].values[-1]

        #Plot test point (horizontal line) and the regression line
        plt.hlines(test_point,xmin, xmax, linestyles='dashed', color='#d7191c', zorder = 5)
        
        pp.savefig(fig)
        plt.close()
        pp.close()

if __name__ == '__main__':
    
    data = calculate_corr_from_total_syanapses('./input/NR_twk-40_gf.xlsx')
    linear, nonlinear, no_trend = filter_by_pvalues(data, 0.05)

    find_neural_age(linear, 'NR')
    inside_outside(no_trend, 'NR')

 
   