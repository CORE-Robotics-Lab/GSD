
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from collections import OrderedDict
from pprint import pprint


def data_mean(arr):
    arr = np.array(arr)
    sidxs = np.argsort(arr)
    # sidxs = sidxs[1:-1]
    return np.mean(arr[sidxs])

def data_std(arr):
    arr = np.array(arr)
    sidxs = np.argsort(arr)
    # sidxs = sidxs[1:-1]
    return np.std(arr[sidxs])

MASTER_COLORS = ['royalblue', 'green', 'red', 'black', 'black', 'black']
def get_master_key(fname, methodkey):

    # mks = ['IG cns=0', 'IG cns=1', 'IG cns=2']
    # mks = ['IA cns=0', 'IA cns=1', 'IA cns=2']

    # mks = ['IG clip=0 off=0', 'IG clip=10 off=1', 'IA clip=0 off=0', 'IA clip=10 off=1']

    # mks = ['IG Vanilla', 'IG-M', 'IG-M + SN', 'IG-M + RR v0', 'IG-M + RR v1']
    # mks = ['IA Vanilla', 'IA-M', 'IA-M + SN', 'IA-M + RR v0', 'IA-M + RR v1']

    # mks = ['IG Vanilla', 'IG-M', 'IG-M + SN', 'IG-M + RR v0', 'IG-M + RR v1', 'IG-M + RR v1 + CNS']
    # mks = ['IA Vanilla', 'IA-M', 'IA-M + SN', 'IA-M + RR v0', 'IA-M + RR v1', 'IA-M + RR v1 + CNS']

    # mks = ['IG', 'IG+M', 'IG+M+SN', 'GSD (Ours)']
    mks = ['IG', 'IG+M', 'IG+M+SN', 'GSD (Ours)', 'IG+M+SN 1', 'GSD (Ours) 1']
    # mks = ['IA', 'IA+M', 'IA+M+SN', 'GSD (Ours)']

    ss = ''
    for mi, mk in enumerate(mks):
        if f'PPO-{mi}' in methodkey or f'PPOBC-{mi}' in methodkey:
            ss += mk

    if fname is not None:
        fs = fname.split('/')[-1].split('.')[0]
        ss = fs[:4] + fs[-2:] + " " + ss

    return ss

def process(fnames):

    NTRAINSEEDS = 5
    # NTRAINSEEDS = 3
    modeltables = OrderedDict()
    for fname in fnames:
        IMODEL = 0
        modeldata = OrderedDict()
        modeltables[fname] = modeldata

        key = None
        result = None

        with open(fname, 'r') as f:
            for line in f:
                if 'Policy model' in line:
                    if key is not None:
                        modeldata[key].append(result)
                        if len(modeldata[key]) >= NTRAINSEEDS:
                            IMODEL += 1
                    ckptstr = line.split(' ')[-1]
                    algo = ckptstr.split('/')[-5]
                    ckptstr = ckptstr.split('/')[-4]
                    dsettokidx = [i for i, x in enumerate(ckptstr.split('_')) if x.startswith('type')][0]
                    dsettok = ckptstr.split('_')[dsettokidx]
                    dsettok = dsettok.split('-')[0]
                    traindset = int(dsettok[4:])
                    ckptseed = int(ckptstr.split('_')[-1][1:])

                    key = f'{algo}-{IMODEL} & {str(traindset)}'
                    if key not in modeldata:
                        modeldata[key] = []

                    result = OrderedDict()
                    result['seed'] = int(ckptseed)

                if line.startswith('RT'):
                    tok, r_mean, r_std, r_max, r_min = line.split(' ')[:5]
                    result[f'{tok}_mean'] = float(r_mean)
                    result[f'{tok}_max'] = float(r_max)
                if line.startswith('LR'):
                    tok, r_mean, r_std, r_max, r_min = line.split(' ')[:5]
                    result[f'{tok}_mean'] = float(r_mean)
                    result[f'{tok}_max'] = float(r_max)
                if line.startswith('VL'):
                    tok, val = line.split(' ')[:2]
                    result[tok] = float(val)
                # if line.startswith('AL'):
                #     tok, val = line.split(' ')[:2]
                #     result[tok] = float(val)
                # if line.startswith('KL'):
                #     tok, val = line.split(' ')[:2]
                #     result[tok] = float(val)
                # if line.startswith('DV'):
                #     tok, val = line.split(' ')[:2]
                #     result[tok] = float(val)

            modeldata[key].append(result)

    # BUDGETS = [10, 40]
    BUDGETS = range(10, 51, 10)
    for fname, modeltable in modeltables.items():
        # print recovery modeltable
        for methodkey, results in modeltable.items():
            for ires in range(len(results)):
                for budget in BUDGETS:
                    tempvals = [modeltables[fname][methodkey][ires][f'VL{budget}-{vid}'] for vid in [1, 3, 5]]
                    modeltables[fname][methodkey][ires][f'VL{budget}-train'] = np.mean(tempvals)
                    tempvals = [modeltables[fname][methodkey][ires][f'VL{budget}-{vid}'] for vid in [2, 4]]
                    modeltables[fname][methodkey][ires][f'VL{budget}-test'] = np.mean(tempvals)

                    tempvals = [modeltables[fname][methodkey][ires][f'RT-VL{budget}-{vid}_mean'] for vid in [1, 3, 5]]
                    modeltables[fname][methodkey][ires][f'RT-VL{budget}-train_mean'] = np.mean(tempvals)
                    tempvals = [modeltables[fname][methodkey][ires][f'RT-VL{budget}-{vid}_mean'] for vid in [2, 4]]
                    modeltables[fname][methodkey][ires][f'RT-VL{budget}-test_mean'] = np.mean(tempvals)

    # assert that all models are covered by master key
    for fname, modeltable in modeltables.items():
        masterkeys = []
        for methodkey in modeltable.keys():
            newmasterkey = get_master_key(fname, methodkey)
            assert newmasterkey not in masterkeys, f"{newmasterkey}, {masterkeys}"
            masterkeys.append(newmasterkey)

    plottable = OrderedDict()
    measurekeys = []
    measurekeys.append('RT_mean')
    measurekeys.append('LR_mean')
    measurekeys.append('VL-std-all')
    for budget in BUDGETS:
        measurekeys.append(f'VL{budget}-all')
        measurekeys.append(f'RT-VL{budget}-all_mean')
        for ivel in [1, 2, 3, 4, 5]:
            measurekeys.append(f'VL{budget}-{ivel}')
            measurekeys.append(f'RT-VL{budget}-{ivel}_mean')
        measurekeys.append(f'VL{budget}-train')
        measurekeys.append(f'RT-VL{budget}-train_mean')
        measurekeys.append(f'VL{budget}-test')
        measurekeys.append(f'RT-VL{budget}-test_mean')
    for mk in measurekeys:
        assert mk not in plottable
        plottable[mk] = OrderedDict()

    for fname, modeltable in modeltables.items():
        # print recovery modeltable
        for methodkey, results in modeltable.items():
            masterkey = get_master_key(fname, methodkey)
            for mk in measurekeys:
                plottable[mk][masterkey] = [dt[mk] for dt in results]

    # latex tables
    from tabulate import tabulate, SEPARATING_LINE
    # interestedkeys = ['RT_mean', 'RT-VL10-all_mean', 'RT-VL40-all_mean']
    # interestedkeys = ['VL10-all', 'VL10-train', 'VL10-test', 'VL40-all', 'VL40-train', 'VL40-test']
    interestedkeys = ['VL10-all', 'VL10-train', 'VL10-test', 'VL50-all', 'VL50-train', 'VL50-test']
    # interestedkeys = ['RT-VL10-all', 'RT-VL10-train', 'RT-VL10-test', 'RT-VL40-all', 'RT-VL40-train', 'RT-VL40-test']
    # interestedkeys = ['VL10-all', 'RT-VL10-all_mean', 'VL40-all', 'RT-VL40-all_mean']
    # interestedkeys = ['RT_mean', 'VL-std-all']
    # interestedkeys = ['VL10-all', 'VL10-train', 'VL10-test', 'VL40-all', 'VL40-train', 'VL40-test']
    # interestedkeys = ['VL-std-all', 'VL10-test', 'VL40-test']
    printabledata = []
    catkey = None
    for mdlk in plottable[interestedkeys[0]].keys():
        newcatkey = mdlk.split(' ')[0]
        if catkey is not None and catkey != newcatkey:
            printabledata.append(SEPARATING_LINE)
        catkey = newcatkey

        printabledatalist = []
        printabledatalist.append(mdlk)
        for mk in interestedkeys:
            vals = plottable[mk][mdlk]
            vstr = f'{np.mean(vals):2.3f} $\pm$ {np.std(vals):2.3f}'
            printabledatalist.append(vstr)
        printabledata.append(printabledatalist)
    print(tabulate(printabledata, ['Model'] + interestedkeys, tablefmt="github"))
    # print(tabulate(printabledata, ['Model'] + interestedkeys, tablefmt="latex_raw"))

    def clog(arr):
        return np.log(np.clip(arr, 1e-8, np.inf))

    print('Curves plotting errors across budgets')
    gridsize=(3, 2)
    plt.rc('font', size=20)
    fig, axes = plt.subplots(*gridsize, figsize=(5*gridsize[1], 5*gridsize[0]))
    for ifile, fname in enumerate(fnames):
        for ikey, evalkey in enumerate(['train', 'test']):
            pltax = axes[ifile][ikey]
            pltax.grid()
            pltax.locator_params(axis='y', nbins=8)

            # only display for first two
            if ifile == 0:
                pltax.set_title(evalkey.capitalize())
            # only display for last two
            if ifile == len(fnames)-1:
                pltax.set_xlabel("K")

            if 'runp' in fname:
                if ikey == 0: pltax.set_ylabel(r"$\bf{" + "Inverted Pendulum" + "}$" + "\nMAE in Slider Pos (m, log)")
                # pltax.ylim(0, 0.08) # pend
                # pltax.ylim(0, 0.16) # pend
            if 'runc' in fname:
                if ikey == 0: pltax.set_ylabel(r"$\bf{" + "HalfCheetah" + "}$" + "\nMAE in Velocity (mps, log)")
                # pltax.set_ylim(-3.5, -1) # hc
            if 'runf' in fname:
                if ikey == 0: pltax.set_ylabel(r"$\bf{" + "FetchPickPlace" + "}$" + "\nMAE in Obj Y-coor (m, log)")
                # pltax.ylim(0, 0.1) # fetch

            interestedkeys = [f'VL{bgt}-{evalkey}' for bgt in range(10, 51, 10)]
            plotticks = [f'{bgt}' for bgt in range(10, 51, 10)]

            # print(plottable[interestedkeys[0]].keys())
            for midx, mdlk in enumerate(plottable[interestedkeys[0]].keys()):
                methodfname, methodlabel = mdlk.split(' ')[0], ' '.join(mdlk.split(' ')[1:])
                if methodfname[:-2] not in fname:
                    continue
                means = np.array([np.mean(plottable[mk][mdlk]) for mk in interestedkeys])
                stds = np.array([np.std(plottable[mk][mdlk]) for mk in interestedkeys])*1
                serr = stds/np.sqrt(5)
                pltax.plot(np.arange(means.shape[0]), clog(means), 'x-', label=methodlabel, markersize=10, color=MASTER_COLORS[midx%4])
                pltax.fill_between(np.arange(means.shape[0]), clog(means-serr), clog(means+serr), alpha=0.2, color=MASTER_COLORS[midx%4])

            pltax.set_xticks(np.arange(means.shape[0]), plotticks)

    handles, labels = pltax.get_legend_handles_labels()
    # pltax.legend(bbox_to_anchor=(0.0, 1.05), ncol=4, fancybox=True)
    legs = fig.legend(handles, labels, loc=(0.00, 0.0), ncol=4, fancybox=True, fontsize=22, markerscale=2)
    for legobj in legs.legendHandles: legobj.set_linewidth(5)
    fig.tight_layout(rect=(0,0.015,1,1))
    fig.savefig('z_curve_error.png')

    print('Curves plotting rewards across budgets')
    gridsize=(1, 3)
    plt.rc('font', size=20)
    fig, axes = plt.subplots(*gridsize, figsize=(5*gridsize[1], 5*gridsize[0]))
    for ifile, fname in enumerate(fnames):
        for ikey, evalkey in enumerate(['all']):
            pltax = axes[ifile]
            pltax.grid()
            pltax.locator_params(axis='y', nbins=8)

            # only display for first
            if ifile == 0:
                pltax.set_ylabel("Episode Reward")
            pltax.set_xlabel("K")

            if 'runp' in fname:
                pltax.set_title(r"$\bf{" + "Inverted Pendulum" + "}$")
            if 'runc' in fname:
                pltax.set_title(r"$\bf{" + "HalfCheetah" + "}$")
            if 'runf' in fname:
                pltax.set_title(r"$\bf{" + "FetchPickPlace" + "}$")

            # if 'runc' in fname:
            #     if ikey == 0: pltax.set_ylabel("MAE in Velocity (mps, log)")
            # if 'runf' in fname:
            #     if ikey == 0: pltax.set_ylabel("MAE in Obj Y-coor (m, log)")

            interestedkeys = [f'RT-VL{bgt}-{evalkey}_mean' for bgt in range(10, 51, 10)]
            plotticks = [f'{bgt}' for bgt in range(10, 51, 10)]

            # print(plottable[interestedkeys[0]].keys())
            for midx, mdlk in enumerate(plottable[interestedkeys[0]].keys()):
                methodfname, methodlabel = mdlk.split(' ')[0], ' '.join(mdlk.split(' ')[1:])
                if methodfname[:-2] not in fname:
                    continue
                means = np.array([np.mean(plottable[mk][mdlk]) for mk in interestedkeys])
                stds = np.array([np.std(plottable[mk][mdlk]) for mk in interestedkeys])*1
                serr = stds/np.sqrt(5)
                pltax.plot(np.arange(means.shape[0]), means, 'x-', label=methodlabel, markersize=10, color=MASTER_COLORS[midx%4])
                pltax.fill_between(np.arange(means.shape[0]), means-serr, means+serr, alpha=0.2, color=MASTER_COLORS[midx%4])

            pltax.set_xticks(np.arange(means.shape[0]), plotticks)

    handles, labels = pltax.get_legend_handles_labels()
    legs = fig.legend(handles, labels, loc=(0.21, 0.0), ncol=4, fancybox=True, fontsize=22, markerscale=2)
    for legobj in legs.legendHandles: legobj.set_linewidth(5)
    fig.tight_layout(rect=(0,0.04,1,1))
    plt.savefig('z_curve_reward.png')

if __name__ == "__main__":
    import sys
    process(sys.argv[1:])
