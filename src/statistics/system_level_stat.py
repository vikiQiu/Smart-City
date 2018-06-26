'''
used to analyze all related things of system
'''
from __future__ import division
import os
import re
import json
import numpy as np
import collections
import operator
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from scipy.ndimage.filters import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.interpolate import interp1d
mpl.rc('font',family='Times New Roman')


def types_temp_analyze(type):
    '''
    used to analyze proportion and timing
    :return:
    '''
    # temporal distribution
    recs=[]
    with open(type+'_time_userID') as f:
        for line in f:
            attrs=line.split(',')
            recs.append(attrs)

    rec_min=collections.defaultdict(list)
    for rec in recs:
        rec_min[int(int(rec[0])/300)].append(rec[1])

    rec_min_final=[0]*288
    for k,v in rec_min.iteritems():
        rec_min_final[k]=len(set(v))
    print rec_min_final
    return rec_min_final

def types_temp_draw3(all_recs):
    all_recs=types_draw
    # maxi=[]
    # for rec in all_recs:
    #     maxi.append(max(rec[1]))
    # print 'Max value of y-axis is %d' %(max(maxi))

    x_ticks=['sRequest','Paging', 'Attach']

    with PdfPages('types_temp3.pdf') as pdf:
        font_size='30'
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.arange(0,288), uniform_filter1d(all_recs[0],2),'r-',marker='D',markersize=5, markevery=7,linewidth=2)
        ax.plot(np.arange(0,288), uniform_filter1d(all_recs[1],2),'b--',marker='o',markersize=5, markevery=7,linewidth=2)
        ax.plot(np.arange(0,288), uniform_filter1d(all_recs[2],2),'g:',marker='s',markersize=5, markevery=7,linewidth=2)


        plt.subplots_adjust(top=0.95, right=0.9, left=0.21, bottom=0.21)
        plt.xticks(np.arange(0,289,step=48),[0,4,8,12,16,20,24],size=font_size,fontweight='bold')
        # plt.yticks(size=font_size,fontweight='bold')
        plt.yticks(np.arange(0,1.1,step=0.2),[0,0.2,0.4,0.6,0.8,1],size=font_size,fontweight='bold')

        plt.yticks(np.arange(0,35001,step=5000),[0,5,10,15,20,25,30,35],size=font_size,fontweight='bold')
        plt.xlabel('Time of Day (h)', fontsize=font_size,fontweight='bold')
        plt.ylabel('# of Records (k)',fontsize=font_size,fontweight='bold')
        # plt.ylim(0,100)
        plt.legend(x_ticks,fontsize=15, loc='lower right')
        plt.xlim(0,289)
        plt.grid()
        plt.show()
        pdf.savefig(fig)

def types_temp_draw4(all_recs):
    all_recs=types_draw
    # maxi=[]
    # for rec in all_recs:
    #     maxi.append(max(rec[1]))
    # print 'Max value of y-axis is %d' %(max(maxi))

    # x_ticks=['LTE_ATTACH','LTE_DETACH','CSFB','Service_Req','TAU','Path_Switch','LTE_PAGING']
    x_ticks=['TAU','CSFB','Detach','pSwitch']

    with PdfPages('types_temp4.pdf') as pdf:
        font_size='30'
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.arange(0,288), gaussian_filter1d(all_recs[0],5),'r-',marker='D',markersize=8, markevery=10,linewidth=2)
        ax.plot(np.arange(0,288), gaussian_filter1d(all_recs[1],5),'b--',marker='o',markersize=8, markevery=10,linewidth=2)
        ax.plot(np.arange(0,288), gaussian_filter1d(all_recs[2],5),'g:',  marker='s',markersize=8, markevery=10,linewidth=2)
        ax.plot(np.arange(0,288), uniform_filter1d(all_recs[3],2),'-.',color='#666666',marker='^',markersize=8, markevery=10,linewidth=2)

        plt.subplots_adjust(top=0.73, right=0.9, left=0.21, bottom=0.21)
        plt.xticks(np.arange(0,289,step=48),[0,4,8,12,16,20,24],size=font_size,fontweight='bold')
        # plt.yticks(size=font_size,fontweight='bold')
        plt.yticks(np.arange(0,35001,step=5000),[0,5,10,15,20,25,30,35],size=font_size,fontweight='bold')

        plt.yticks(size=font_size,fontweight='bold')
        plt.xlabel('Time of Day (h)', fontsize=font_size,fontweight='bold')
        plt.ylabel('# of Records (k)',fontsize=font_size,fontweight='bold')
        # plt.ylim(0,100)
        plt.legend(x_ticks,   fontsize=30, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.6) ,columnspacing=0.01, handletextpad=0.1)
        plt.xlim(0,289)
        plt.grid()
        plt.show()
        pdf.savefig(fig)

def types_user_num():
    '''
    used to count number of user
    :return:
    '''
    total_count=[]
    with open('user_service_type_num_dist') as f:
        for line in f:
            attrs=line.split(',')
            total_count.append(int(attrs[2]))
    final_count=dict(collections.Counter(total_count))
    y_vals1=[];total=0
    for k,v in final_count.iteritems():
        y_vals1.append([int(k),int(v)])
        total+=int(v)
    y_vals1.sort(key=lambda x: x[0])
    y_vals=[]
    for item in y_vals:
        y_vals.append(item[1]/total)

    y_vals=[]
    for num in total_count:
        y_vals.append(num/sum(total_count))

    with PdfPages('types_user_num.pdf') as pdf:
        font_size='30'
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.bar(np.arange(1,8), [100*v for v in y_vals], align='center',color='#666666')

        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            value = height
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%0.2f' % value,
                    ha='center', va='bottom',fontsize=20)

        plt.subplots_adjust(top=0.95, right=0.9, left=0.21, bottom=0.21)
        plt.xticks(np.arange(1,8),size=font_size,fontweight='bold')
        # plt.yticks(size=font_size,fontweight='bold')
        plt.yticks(np.arange(0,51,step=10),[0,10,20,30,40,50],size=font_size,fontweight='bold')

        plt.yticks(size=font_size,fontweight='bold')
        plt.xlabel('# of signaling types',fontsize=font_size,fontweight='bold')
        plt.ylabel('Percent of users (%)',fontsize=font_size,fontweight='bold')
        # plt.ylim(0,100)
        # plt.legend(y_ticks,fontsize=15, loc='upper left')
        # plt.xlim(0,86401)
        # plt.grid()
        plt.show()
        pdf.savefig(fig)

def types_user_dist():
    type_cell_num=collections.defaultdict(int)
    count=0
    with open('user_service_type_num_dist') as f:
        for line in f:
            count+=1
            attrs=line.split(',')
            if int(attrs[2])==1:
                type_cell_num[attrs[-2]]+=1
            elif int(attrs[2])==2:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
            elif int(attrs[2])==3:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
                type_cell_num[attrs[-6]]+=1
            elif int(attrs[2])==4:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
                type_cell_num[attrs[-6]]+=1
                type_cell_num[attrs[-8]]+=1
            elif int(attrs[2])==5:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
                type_cell_num[attrs[-6]]+=1
                type_cell_num[attrs[-8]]+=1
                type_cell_num[attrs[-10]]+=1
            elif int(attrs[2])==6:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
                type_cell_num[attrs[-6]]+=1
                type_cell_num[attrs[-8]]+=1
                type_cell_num[attrs[-10]]+=1
                type_cell_num[attrs[-12]]+=1
            elif int(attrs[2])==7:
                type_cell_num[attrs[-2]]+=1
                type_cell_num[attrs[-4]]+=1
                type_cell_num[attrs[-6]]+=1
                type_cell_num[attrs[-8]]+=1
                type_cell_num[attrs[-10]]+=1
                type_cell_num[attrs[-12]]+=1
                type_cell_num[attrs[-14]]+=1
    y_vals1=[]
    y_ticks=[]
    for k,v in type_cell_num.iteritems():
        print 'Type name: %s, \t number of users: %d' %(k,v)
        y_ticks.append(k)
        y_vals1.append(v)
    y_vals=[100*v/count for v in y_vals1]
    print y_vals
    print sum(y_vals1)
    print 'number of users: %d' %count

    # y_vals=[90.90, 5.30, 9.20, 24.38, 35.52, 23.55, 83.62]
    # y_vals=sorted(y_vals,reverse=True)
    y_ticks= ['TAU','Detach', 'CSFB','pSwitch', 'Attach',  'Paging', 'sRequest']

    with PdfPages('types_user_dist.pdf') as pdf:
        font_size='30'
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.bar(np.arange(1,8), y_vals, align='center',color='#666666')

        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            value = height
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%0.2f' % value,
                    ha='center', va='bottom',fontsize=20)

        plt.subplots_adjust(top=0.95, right=0.9, left=0.21, bottom=0.36)
        plt.xticks(np.arange(1,8),y_ticks,size=font_size,fontweight='bold',rotation=90)
        plt.yticks(np.arange(0,101,step=20),[0,20,40,60,80,100],size=font_size,fontweight='bold')

        # plt.xticks(np.arange(0,8,step=1),y_ticks, size=font_size,fontweight='bold')
        # plt.xlabel('Signaling service type',fontsize=font_size,fontweight='bold')
        plt.ylabel('% of users',fontsize=font_size,fontweight='bold')
        plt.ylim(0,105)
        # plt.legend(y_ticks,fontsize=15, loc='upper right')
        # plt.xlim(0,86401)
        # plt.grid()
        plt.show()
        pdf.savefig(fig)

def user_rec_analyze():
	'''
	distribution of number of service types of each user
	:return:
	'''
	recs=[]
	data_path='/Users/qzhit/Google Drive/hefei_data/hefei_codes/qz_hf_ubuntu_codes/HefeiCodes/sampleSignalAll/'
	files=[]
	try:
		for j in os.listdir(data_path):
			if j.startswith('part'):
				files.append(open(data_path+j))
	except OSError:
		pass

	# record number of service types a user has
	user_sevice_types=collections.defaultdict(list)
	for i in files:
		for line in i:
			try:
				attrs=line.split(',')
				user_sevice_types[attrs[3]].append(attrs[-2])#userID, service type
			except:
				pass

	with open('user_service_type_num_dist','w') as f:
		for k,v in user_sevice_types.iteritems():# k: userID, v: service types list
			count=len(v)# number of records
			types_count_dict=dict(collections.Counter(v))# a dict
			line=''
			for k1,v1 in types_count_dict.iteritems():
				line+=','+k1+','+str(v1)
			types_count=len(set(v))# number of how many types does this user have
			line=k+','+str(count)+','+str(types_count)+line+'\n'
			f.write(line)

def cdf_rec_num_user():
    # cdf of number of records per user
    rec_num=[]
    with open('user_service_type_num_dist') as f:
        for line in f:
            attrs=line.split(',')
            rec_num.append(int(attrs[1]))

    rec_num.sort()

    y_vals1=[]
    x_vals1=[]
    rec_num_dict=dict(collections.Counter(rec_num))
    count=0
    for k,v in rec_num_dict.iteritems():
        x_vals1.append(int(k))
        y_vals1.append(count)
        count+=v

    y_vals2=[100*k/count for k in y_vals1]
    count1=0;x_vals=[];y_vals=[]
    for i in range(len(x_vals1)):
        if x_vals1[i]<=1000:
            count+=1
            x_vals.append(x_vals1[i])
            y_vals.append(y_vals2[i])

    print 'Number of data cut: %f' %(100*(len(y_vals)/len(y_vals2)))
    print 'Number of users covered: %f' %(max(y_vals))

    with PdfPages('user_rec_num_cdf.pdf') as pdf:
        font_size='30'
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(x_vals, uniform_filter1d(y_vals,3),'-',linewidth=4)

        plt.subplots_adjust(top=0.95, right=0.9, left=0.21, bottom=0.21)
        # plt.xticks(size=font_size,fontweight='bold')
        plt.yticks(np.arange(0,101,step=20),[0,20,40,60,80,100],size=font_size,fontweight='bold')

        plt.xticks(np.arange(0,1001,step=200),[0,2,4,6,8,10],size=font_size,fontweight='bold')
        plt.xlabel('# of Records (100)', fontsize=font_size,fontweight='bold')
        plt.ylabel('% of Users',fontsize=font_size,fontweight='bold')
        plt.grid()
        plt.show()
        pdf.savefig(fig)

if __name__ == '__main__':
    # user_count()

    types=['LTE_ATTACH','LTE_DETACH','CSFB','Service_Req','TAU','Path_Switch','LTE_PAGING']

    user_rec_analyze()



