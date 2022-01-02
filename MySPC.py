import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm

def hist(data, label, lsl, usl, cp, ppk=0):
    # data - Array containing the data to display
    # label - label to use in the title
    # lsl, usl - Lower and Upper Service Limits
    # cp - theoretical process capability used to generate the data
    
    # Calculate number of Histogram bins using Sturge’s Rule
    # as per: https://www.statisticshowto.com/choose-bin-sizes-statistics/
    bins = int(np.round(np.log10(len(data))*3.322+1))

    fig, ax = plt.subplots(1, 1)
    
    # Create a histogram, capture count to calculate position of the USL and LCL labels
    count, bins, patches = ax.hist(data, bins, density=False, label=label, alpha=0.8)
    
    max_y = max(count)
    
    # Draw red vertical lines for upper and lower tolerances
    ax.axvline(usl, color='r', linestyle='dashed', linewidth=1)
    ax.axvline(lsl, color='r', linestyle='dashed', linewidth=1)
    # Add labels to the vertical lines
    # https://stackoverflow.com/questions/13413112/creating-labels-where-line-appears-in-matplotlib-figure
    # y position of the label calculated using the count of the hihgest bin and divided by 10
    ax.text(lsl, max_y/10, 'LSL',rotation=90)
    ax.text(usl, max_y/10, 'USL',rotation=90)
    
    # Draw Probability density function for normal distribution
    
    # Get limit of the X axis
    # https://stackoverflow.com/questions/26131607/matplotlib-get-ylim-values
    x_min, x_max = ax.get_xlim()
    
    # https://realpython.com/how-to-use-numpy-arange/
    x = np.arange(x_min, x_max, (x_max-x_min)/500. )
    
    # calculate PDF using scipi.stats package:
    y = norm.pdf(x, loc=data.mean(), scale=data.std())
    
    # calculate the max bin count to max pdf ration to use it as scaling factor for pdf:
    scale = max_y/max(y)
    
    # Plot pdf
    ax.plot(x, y*scale)
    
    # https://saralgyaan.com/posts/matplotlib-tutorial-in-python-chapter-5-filling-area-on-line-plots/
    ax.fill_between(x, 0, y*scale, color='red', where=(x<lsl), alpha=0.7 )
    ax.fill_between(x, 0, y*scale, color='red', where=(x>usl), alpha=0.7 )

    # define plot's parameters
    plt.rcParams["figure.figsize"] = [8, 4]
    plt.xlabel('Mmeasurements')
    plt.ylabel('Frequency of occurence')
    if ppk==0:
        plt.title('Distribution of {} in a process with Cp={:.2}'.format(label, cp))
    else:
        plt.title('Distribution of {} in a process with Cp={:.2} and Ppk={:.2}'.format(label, cp, ppk))
    plt.legend(['LSL', 'USL', 'PDF', label])
    
    return fig, ax
    
    
    
    
# Define a function that will check if datapoints are violating any of the SPC rules
def spc_stats(dim, std_in=[], n=1):
    
    # Define DataFrame with one column 'Dim'
    data = pd.DataFrame({'Dim': dim})
    
    # Add column with moving range:
    # Calculated as per https://stackoverflow.com/questions/30673209/pandas-compare-next-row
    data['mR'] = data['Dim']-data['Dim'].shift(1)    
    
    # Add 'Reason' column
    data['Reason'] = 0    
    
    # use calculations for X-bar S chart if n>1
    if n>1:
        # Calculations for below parameters as per 'Implementing Six Sigma' Forrest W. Breyfogle III, p1059
        
        c4 = 4*(n-1)/(4*n-3)
        a3 = 3/(c4*np.sqrt(n))                
        b3 = 1-3/(c4*np.sqrt(2*(n-1)))
        b4 = 1+3/(c4*np.sqrt(2*(n-1))) 
        
        # use standard deviation array instead of moving range 
        data['variability'] = std_in
        
        #calculate mean of X-bar and standard deviation
        x_bar = np.mean(data['Dim'])
        mr_bar = np.mean(std_in)
        
        #calulcate upper and lower contol limits for X-bar and standard deviation
        ucl = x_bar+mr_bar*a3
        lcl = x_bar-mr_bar*a3
        
        mr_ucl = mr_bar*b4
        mr_lcl = mr_bar*b3
                                
        
    # use calculations for X mR chart
    else:
        # Set variability to moving range
        data['variability'] = data['mR']
    
        # calculate the mean of the variable
        x_bar = np.mean(data['Dim'])
        mr_bar = np.mean( np.abs(data['mR'][1:len(data['mR'])]))

        # Below calculations as per 'Implementing Six Sigma' Forrest W. Breyfogle III, p227
        # Calculate Upper and Lower Control Limits for X chart 
        ucl = x_bar+mr_bar*2.66
        lcl = x_bar-mr_bar*2.66

        # Calculate Upper Control Limits for mR chart (there is no lower CL for mR chart)
        mr_ucl = mr_bar+mr_bar*3.267          
    
    # Shewhart SPC control chart rules:
    # https://analyse-it.com/docs/user-guide/process-control/shewhart-control-chart-rules
    
    # Number of observation on the same side of the mean when the alarm is switched on (typically 8 or 9)
    n_side=8
    
    # Number of consecutive points steadily increasing or decreasing to switch the alarm (typically 6)
    n_drift = 6
    
    # Number of consecutive points are alternating up and down (typically 14)
    n_alter = 8
    
    # Initiate 'Reason' column with zeros
    # Each violation of the SPC rules will be added as a power of 2. At the end, 'Reason' column will be converted to binary string
    # This way, I can detect more than one SPC rule violation
    data['Reason'] = 0
    
    # Check for measurements aoutside the contro limits
    for index, row in data.iterrows():
        x = row[0]
        mr = row[1]
        
        if x>ucl or x<lcl:
            data.loc[index, 'Reason'] += 1
            
        if mr>mr_ucl:
            data.loc[index, 'Reason'] += 2
            
    
    # check if there ar at least 'n_side' points on the same side of the mean line
    
    # 1 if point is above x_bar, -1 if it's below
    data['xbar_side'] = np.sign(data['Dim'] - x_bar)

    # count running sum of last n_side ['xbar_side'] values
    data['sameside'] = data['xbar_side']
    for i in range(1, n_side):
        data['sameside'] =  data['sameside']  + data['xbar_side'].shift(i)
    
    data.loc[ np.abs(data['sameside'])==n_side , 'Reason'] += 4
    
    # Check for drifts in data: points are steadily increasing or decreasing
    
    data['delta_sign']=np.sign( data['mR'] )
    
    # count running sum of last n_side ['xbar_side'] values
    data['drift'] = data['delta_sign']
    for i in range(1, n_drift):
        data['drift'] =  data['drift'] + data['delta_sign'].shift(i)
    
    data.loc[ np.abs(data['drift'])==n_drift , 'Reason'] += 8
    
    # check for consecutive points are alternating up and down
    
    # column 'alter' will be 0 if 2 consecutive points alter in direction of change
    # and will be 2 if they both going up or down
    data['alter'] = np.abs(data['delta_sign'] + data['delta_sign'].shift(1) )
    
    # if there sum of the last n_alter rows of 'alter' column equal 0
    # Then that means that these points were alternating up and down for n_alter consecutive points
    data['sum_alter'] = data['alter']
    for i in range(1, n_alter):
        data['sum_alter'] =  data['sum_alter'] + data['alter'].shift(i)
    
    data.loc[ np.abs(data['sum_alter'])==0 , 'Reason'] += 16    
    
    # Translate the sum of reasons into binary string
    # as per: https://stackoverflow.com/questions/45018601/decimal-to-binary-in-dataframe-pandas
    data['Reason'] = data.Reason.apply(lambda x: format(int(x), '05b'))
    
    if n==1:
        return data[['Dim', 'variability', 'Reason']], x_bar, mr_bar, ucl, lcl, mr_ucl
    else:
        return data[['Dim', 'variability', 'Reason']], x_bar, mr_bar, ucl, lcl, mr_ucl, mr_lcl
    
    
# Basis for this code taken from: https://towardsdatascience.com/quality-control-charts-guide-for-python-9bb1c859c051
def xmr(data_in, n=0):
    # data_in - Array containing the data to display
    # n - show only last n measurements, if n=0, show all
    
    # Make sure n is greater than 0 and smaller than the lenght of the data array
    if n>0 and n<len(data_in):
        dim = data_in[-n:]
    else:
        dim = data_in
    
    # Find the points that are violating SPC rules and calculate the means and control limits
    data, x_bar, mr_bar, ucl, lcl, mr_ucl = spc_stats(dim)


    # Plot x and mR charts
    fig, axs = plt.subplots(2, figsize=(8,8), sharex=True)

    # X chart
    # Graph all the points 
    axs[0].plot(data['Dim'], linestyle='-', marker='o', color='black')

    
    # Add red dot when there are too many points on the same side of the mean
    axs[0].plot(data[data['Reason'].str[-5]=="1"]['Dim'], linestyle="", marker='o', color='blue')
    
    # Add red dot when there are too many points on the same side of the mean
    axs[0].plot(data[data['Reason'].str[-4]=="1"]['Dim'], linestyle="", marker='o', color='yellow')    
    
    # Add red dot when there are too many points on the same side of the mean
    axs[0].plot(data[data['Reason'].str[-3]=="1"]['Dim'], linestyle="", marker='o', color='orange')    
    
    # Add red dot where Dim is over UCL or under LCL
    axs[0].plot(data[data['Reason'].str[-1]=="1"]['Dim'], linestyle="", marker='o', color='red')    

    # Plot blue horizontal line at the process mean
    axs[0].axhline(x_bar, color='blue')

    # Plot red dotted lines at UCL and LCL
    axs[0].axhline(ucl, color = 'red', linestyle = 'dashed')
    axs[0].axhline(lcl, color = 'red', linestyle = 'dashed')

    # Set Chart title and axis labels
    axs[0].set_title('Individual Chart')
    axs[0].set(xlabel='Part', ylabel='Measurement')


    # mR chart
    # Graph all the points 
    axs[1].plot( np.abs(data['variability']), linestyle='-', marker='o', color='black')

    # Add red dot where Dim is over UCL
    axs[1].plot(data[data['Reason'].str[-2]=="1"]['variability'], linestyle="", marker='o', color='red')

    # Plot blue horizontal line at the mR mean
    axs[1].axhline(mr_bar, color='blue')

    # Plot red dotted line at UCL
    axs[1].axhline(mr_ucl, color='red', linestyle ='dashed')

    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Moving Range Chart')
    axs[1].set(xlabel='Part', ylabel='Range')
    
    return data


# Basis for this code taken from: https://towardsdatascience.com/quality-control-charts-guide-for-python-9bb1c859c051
def xBarS(data_in, col, n=0, Show=True):
    # data_in - Pandas DataFrame containing the data to display
    # col - column from the data_in table to be displayed
    # n - optional, number of last datapoint to display; display all if not provided
    
    # specify average number of parts in lot, as the UCL and LCL values will depend on this
    avg_parts_in_lot = len(data_in)/len(data_in['Lot'].unique())
    
    # Calculate mean value pfor each lot
    dim = data_in[['Lot', col]].groupby('Lot').mean()
    std = data_in[['Lot', col]].groupby('Lot').std()
    
    # Find the points that are violating SPC rules and calculate the control limits
    alldata, x_bar, mr_bar, ucl, lcl, mr_ucl, mr_lcl = spc_stats(dim[col], std[col], avg_parts_in_lot)    
    
    # Plot the graphs only if Show = True
    # It is true by default, but it can be switched off. This function fill only return the points that triggered
    # the SPC Rules and will not plot anything
    if(Show):
    
        # Make sure n is greater than 0 and smaller than the lenght of the data array
        if n>0 and n<len(dim):
            # Trim the number of displayed poinints to the last n:
            data = alldata[-n:]
            #dim = dim[-n:]    

        # Plot x and mR charts
        fig, axs = plt.subplots(2, figsize=(8,8), sharex=True)

        # X chart
        # Graph all the points 
        axs[0].plot(data['Dim'], linestyle='-', marker='o', color='black')


        # Add red dot when there are too many points on the same side of the mean
        axs[0].plot(data[data['Reason'].str[-5]=="1"]['Dim'], linestyle="", marker='o', color='blue') 

        # Add red dot when there are too many points on the same side of the mean
        axs[0].plot(data[data['Reason'].str[-4]=="1"]['Dim'], linestyle="", marker='o', color='yellow')    

        # Add red dot when there are too many points on the same side of the mean
        axs[0].plot(data[data['Reason'].str[-3]=="1"]['Dim'], linestyle="", marker='o', color='orange')    

        # Add red dot where Dim is over UCL or under LCL
        axs[0].plot(data[data['Reason'].str[-1]=="1"]['Dim'], linestyle="", marker='o', color='red')    

        # Plot blue horizontal line at the process mean
        axs[0].axhline(x_bar, color='blue')

        # Plot red dotted lines at UCL and LCL
        axs[0].axhline(ucl, color = 'red', linestyle = 'dashed')
        axs[0].axhline(lcl, color = 'red', linestyle = 'dashed')

        # Set Chart title and axis labels
        axs[0].set_title('X-Bar Chart for {}'.format(col))
        axs[0].set(xlabel='Lot number', ylabel='Measurement')


        # mR chart
        # Graph all the points 
        axs[1].plot( np.abs(data['variability']), linestyle='-', marker='o', color='black')

        # Add red dot where Dim is over UCL
        axs[1].plot(data[data['Reason'].str[-2]=="1"]['variability'], linestyle="", marker='o', color='red')

        # Plot blue horizontal line at the mR mean
        axs[1].axhline(mr_bar, color='blue')

        # Plot red dotted line at UCL and LCL
        axs[1].axhline(mr_ucl, color='red', linestyle ='dashed')
        axs[1].axhline(mr_lcl, color='red', linestyle ='dashed')

        axs[1].set_ylim(bottom=0)
        axs[1].set_title('Standard deviation Chart for {}'.format(col))
        axs[1].set(xlabel='Lot Number', ylabel='Standard deviation')
    
    # return all the data points: lot with respective average, standard deviation within a lot and the violated SPC rules (if any)
    # Use all the data points, not only the ones that are trmmed for display
    return alldata



# Define a function to plot 2 normal distributions pdf side by side for 2 samples
def show_mean_diff(sample1, sample0, sig, det, tValue, det_dict=[]):
    fig, ax = plt.subplots(figsize=(8,4))

    #calculate mean values of sample 1 and sample 2
    on_mean = sample1.mean()
    off_mean = sample0.mean()

    #calculate standard variation of sample 1 and 2
    on_std = sample1.std()
    off_std = sample0.std()

    # calculate the extent of the X axis
    # making sure that 99.9% of both distributions will be included
    x_min = min(norm.interval(0.999, loc=on_mean, scale=on_std)[0], norm.interval(0.999, loc=off_mean, scale=off_std)[0])
    x_max = max(norm.interval(0.999, loc=on_mean, scale=on_std)[1], norm.interval(0.999, loc=off_mean, scale=off_std)[1])

    # generate 500 x values using x_min and x_max established above
    x = np.linspace(x_min, x_max, 500)

    # calculate the y values for 1st sample using normal distribution pdf
    y_on = norm.pdf(x, loc=on_mean, scale=on_std)
    # plot the first samples pdf
    ax.plot(x, y_on, '-g',label='{} On'.format(det))
    # add a vertical line for the 1st samples mean
    ax.axvline(on_mean, color='g', linestyle='dashed', linewidth=1)

    # calculate the y values for 2nd sample using normal distribution pdf
    y_off = norm.pdf(x, loc=off_mean, scale=off_std)
    # plot the second samples pdf
    ax.plot(x, y_off, '-r', label='{} Off'.format(det))
    # add a vertical line for the 2nd samples mean
    ax.axvline(off_mean, color='r', linestyle='dashed', linewidth=1)
    
    # Add a rectangle displaying the difference in the means and calculated p-Value
    # With an arrow pointing at the top of the sample 1 pdf
    desc = ax.annotate(r'$\Delta$ in means ='+'{:.3f} \n p-Value={:.3f}'.format(on_mean-off_mean, tValue), # text to be displayed in the on-plot rectangle
                   xy=(on_mean, max(y_on)), xycoords= 'data',                                              # coordinates of the arrow: top of the sample 1 pdf
                   xytext=(0.7, 0.6), textcoords= 'axes fraction',                                         # coordinates of the text (using relative size of the axes)
                   arrowprops=dict(facecolor='black',  arrowstyle="->", connectionstyle="arc3"),           # style of the arrow
                   horizontalalignment='left', verticalalignment='bottom',                                 # anchor point text
                   fontsize=11,                                                                            
                   bbox=dict(boxstyle="round", fc="w", alpha=0.7)                                          # Stile of the box: rounded edges, white, 30% transparency
                     )
    
    # Add a title of th eplot
    if len(det_dict)>0:
        ax.set_title("{} detected by {}".format(sig, det_dict[det]))
    else:
        ax.set_title("{} split by {}".format(sig, det))

    # add a legend
    ax.legend()

    plt.show()