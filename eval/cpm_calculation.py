# file name format = 'model name + epoch + subset number' ex) Inception-V3_1_Epoch_Subset_0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_sensitivity(df, fps, scans):
    
    def _threshold():
        nonlocal df, fps, scans
        ndf = df[df['y']==0].sort_values(by='acc_1', ascending=0)
        ndf['idx'] = [i for i in range(1, ndf.shape[0]+1)]
        ndf = ndf.set_index('idx')
        fp = np.array(fps) * scans
        threshold = [ndf['acc_1'][int(_)] for _ in fp]
        return threshold
    
    def _y_hat(acc_1, threshold):
        hypothesis = 0
        if acc_1 >= threshold:
            hypothesis = 1
        return hypothesis

    def _cfmat():
        nonlocal df
        confusion_matrix = []  # tn, fp, fn, tp
        for i in [0, 1]:
            for j in [0, 1]:
                count = df['y'][(df['y']==i)&(df['y_hat']==j)].count()
                confusion_matrix.append(count)
        return confusion_matrix
    
    threshold = _threshold()
    sensitivity = []
    for th in threshold:
        df['y_hat'] = df['acc_1'].apply(lambda x: _y_hat(x, th))
        _, _, fn, tp = _cfmat()
        sensitivity.append(tp / (tp + fn))
    return sensitivity

def get_cpm(sample, fps, model_name, epoch):

    total_number_of_subset = 10
    tmp=np.zeros(shape=len(fps))

    for subset_number in range(total_number_of_subset):

        sample = pd.read_csv(str(model_name)+'_'+str(epoch)+'_Epoch_Test_'+str(subset_number)+'.csv') # modification

        if subset_number < 8:
            scans = 89
        else:
            scans = 88

        sensitivity = get_sensitivity(sample, fps, scans)

        print("="*80)
        print('sensitivity of subset_'+str(subset_number)+' On '+str(model_name))
        print(sensitivity)
        print("="*80)
        print("\n")

        tmp += sensitivity

    average_sensitivity = tmp/total_number_of_subset
    print('='*80)
    print('average sensitivity of total subset On '+str(model_name))
    print(average_sensitivity)
    print('='*80)
    print("\n")

    cpm = np.zeros(shape=1)

    for _ in range(len(fps)):
        cpm += average_sensitivity[_]
    cpm = cpm / len(fps)

    print('='*70)
    print(str(model_name)+', '+str(epoch)+' Epoch CPM(Competition Performance Metric) is %.3f' %(cpm))
    print('='*70)

    return average_sensitivity, cpm

def cpm_artist(fps,epoch,models,sensi_storage,cpm_storage):
    fps_interval = np.linspace(start=min(fps), stop=max(fps), num=len(fps), endpoint=True, dtype='float32')
    sensitivity_interval = np.linspace(start=0.0, stop=1.0, num=11, endpoint=True, dtype='float32')

    y_axis_min = 0.0
    y_axis_max = 1.0

    title_fontsize = 11
    axis_fontsize = 9
    color_box = ['k','r','g','b','m','y'] # black, red, green, blue, magenta, yellow

    for index in range(len(models)):
        cpm_storage[index] = '%.3f'%cpm_storage[index]
        plt.plot(fps, sensi_storage[index], color_box[index], linewidth=2.0)
        plt.xticks(fps_interval, fps)
        plt.yticks(sensitivity_interval)
        plt.xlim(left=min(fps), right=max(fps))
        plt.ylim(bottom=y_axis_min, top=y_axis_max)
        plt.title('FROC performance', fontsize=title_fontsize)
        plt.xlabel('Average number of false positives per scan', fontsize=axis_fontsize)
        plt.ylabel('Sensitivity', fontsize=axis_fontsize)
        plt.grid(color='k', linestyle=':', linewidth=0.5) # k is black

    legend = plt.legend([
        str(models[0])+', CPM = '+str(cpm_storage[0]),
        str(models[1])+', CPM = '+str(cpm_storage[1]),
        str(models[2])+', CPM = '+str(cpm_storage[2]),
        str(models[3])+', CPM = '+str(cpm_storage[3]),
        str(models[4])+', CPM = '+str(cpm_storage[4]),
        str(models[5])+', CPM = '+str(cpm_storage[5])],
        loc='lower right',frameon=True)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_linewidth(0.7)

    ## choose either plt.show() or plt.savefig()
    #plt.show()
    plt.savefig(str(epoch)+' Epoch FROC performance.png')
    
if __name__ == '__main__':
    
    fps = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    epoch = 10
    subset_start_number = 0 # fixed
    sensi_storage = []
    cpm_storage = []
    models = ['LeNet-5', 'VGG-16', 'Inception-V3', 'ResNet-152', 'DenseNet-201', 'NASNet'] #fixed

    for index in range(len(models)):

        model_name = models[index]

        sample = pd.read_csv(str(model_name)+'_'+str(epoch)+'_Epoch_Test_'+str(subset_start_number)+'.csv')
        avg_sensitivity, cpm = get_cpm(sample, fps, model_name, epoch)
        sensi_storage.append(avg_sensitivity)
        cpm_storage.append(cpm)

    cpm_artist(fps,epoch,models,sensi_storage,cpm_storage)

