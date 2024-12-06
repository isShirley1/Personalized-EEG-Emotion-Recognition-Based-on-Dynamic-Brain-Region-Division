import datetime
import copy
import os.path as osp
from train_model import *
from utils import *
import itertools

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        result_path = osp.join('./save/', 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,"results_{}.txt".format(args.dataset_name))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(LPEEGNet) + ' on ' + str(args.dataset_name) +
                   "\n1)depth:" + str(args.depth) +
                   "\n2)num-heads:" + str(args.num_heads) +
                   "\n3)resolution:" + str(args.resolution) +
                   "\n4)min_size:" + str(args.min_size) +
                   "\n5)n-subgraph:" + str(args.n_subgraph) +
                   "\n6)GCN-layer-num:" + str(args.GCN_layer_num) +
                   "\n7)max-epoch:" + str(args.max_epoch)+
                   "\n8)max-epoch-cmb:" + str(args.max_epoch_cmb)+
                   "\n9)n-splits1:" + str(args.n_splits1) +
                   "\n10)label-type:" + str(args.label_type) +
                   "\n11)emb-size:" + str(args.emb_size) +
                   "\n12)segment-length:" + str(args.segment_length) +
                   "\n13)overlap:" + str(args.overlap) +
                   "\n14)dropout-MHAttention:" + str(args.dropout_MHAttention) +
                   "\n15)dropout-FC:" + str(args.dropout_FC) +
                   "\n16)dropout-LPEEGNet:" + str(args.dropout_LPEEGNet)  +
                   "\n17)margin:" + str(args.margin) +
                   "\n18)learning-rate:" + str(args.learning_rate)+
                   "\n19)batch-size:" + str(args.batch_size) +
                   "\n20)input-montage:" + str(args.input_montage)+
                   "\n21)temperature:" + str(args.temperature)+
                   "\n22)optimizerM:" + str(args.optimizerM)+
                   "\n23)weight-decay:" + str(args.weight_decay)+
                   "\n24)momentum:" + str(args.momentum)+
                   "\n25)random-seed:" + str(args.random_seed)+"\n")
        file.close()

        result_path_pro = osp.join('./save/', 'process')
        ensure_path(result_path_pro)
        self.text_file_pro = osp.join(result_path_pro, "process_{}.txt".format(args.dataset_name))
        file_pro = open(self.text_file_pro, 'a')
        file_pro.write("\n" + str(datetime.datetime.now()) +"\n")
        file_pro.close()


    def prepare_data(self, idx_train, idx_test, data, label):
        label = np.squeeze(label)
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        data_train, data_test = self.normalize(train=data_train, test=data_test)
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).float()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test


    def normalize(self, train, test):
        n, channels, DE_num, emb = train.shape
        train_data = train.reshape((n, channels, DE_num*emb))
        for channel in range(train.shape[1]):
            mean = np.mean(train[:, channel, :])
            std = np.std(train[:, channel, :])
            train[:, channel, :] = (train[:, channel, :] - mean) / std
            test[:, channel, :] = (test[:, channel, :] - mean) / std
        train = train_data.reshape((n, channels, DE_num, emb))
        return train, test


    def TrailKFold(self, data, chan_in_area, Label, DC, n_splits=2, shuffle=False):
        n = data.shape[0]
        num_segment = self.args.num_segment
        trial = int(n//num_segment)
        k = trial // n_splits
        Trials = list(range(trial))
        if shuffle:
            random.shuffle(Trials)
        splits_idx = []
        splits_chan_idx = []
        Label_idx = []
        DC_idx = []
        idx_train = []
        idx_test = []
        chan_train = []
        chan_test = []
        Label_train = []
        Label_test = []
        DC_train = []
        DC_test = []
        for i in range(n_splits):
            select = Trials[i * k:(i + 1) * k] if i < (n_splits - 1) else Trials[i * k:]
            idx = []
            for j in select:
                idx.extend(list(range(j, j + num_segment)))
            splits_chan_idx.append(chan_in_area[0][select])
            Label_idx.append(Label[select])
            DC_idx.append([DC[i] for i in select])
        for h in range(n_splits):
            if h == 0:
                idx_train.append(list(itertools.chain(*splits_idx[h + 1:])))
                chan_train.append(np.concatenate(splits_chan_idx[h + 1:]).reshape(1, -1))
                Label_train.append(np.array(list(itertools.chain(*Label_idx[h + 1:])),dtype=object))
                DC_train.append(list(itertools.chain(*DC_idx[h + 1:])))
            elif h == (n_splits - 1):
                idx_train.append(list(itertools.chain(*splits_idx[0:h])))
                chan_train.append(np.concatenate(splits_chan_idx[0:h]).reshape(1, -1))
                Label_train.append(np.array(list(itertools.chain(*Label_idx[0:h])),dtype=object))
                DC_train.append(list(itertools.chain(*DC_idx[0:h])))
            else:
                idx_train.append(list(itertools.chain(*splits_idx[0:h], *splits_idx[h + 1:])))
                a = np.concatenate(splits_chan_idx[0:h])
                b = np.concatenate(splits_chan_idx[h + 1:])
                chan_train.append(np.concatenate((a, b)).reshape(1, -1))
                Label_train.append(np.array(list(itertools.chain(*Label_idx[0:h], *Label_idx[h + 1:])),dtype=object))
                DC_train.append(list(itertools.chain(*DC_idx[0:h], *DC_idx[h + 1:])))
            idx_test.append(splits_idx[h])
            chan_test.append(splits_chan_idx[h].reshape(1, -1))
            Label_test.append(Label_idx[h])
            DC_test.append(DC_idx[h])
        return zip(idx_train, idx_test, chan_train, chan_test, Label_train, Label_test, DC_train, DC_test)


    def n_fold_CV(self, subject=[0], fold=10, shuffle=True):
        tta = []
        tva = []
        ttf = []
        tvf = []
        pathname = r'/root/autodl-tmp/TRIAL/Louvain_DEAP'

        TIMER_ = Timer()
        for sub in subject:
            data, label = load_data(self.args.dataset_name, 'extracted', sub + 1)
            label = np.squeeze(label)
            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            filename = pathname + '/sub' + str(sub+1) + '.mat'
            chan_in_area = scio.loadmat(filename)['Louvain']
            save_name = str(sub + 1) + 'degree_centrality.json'
            save_path = os.path.join(pathname, save_name)
            with open(save_path, 'r') as f:
                DC = json.load(f)
            TIMER = Timer()
            for idx_fold, (idx_train, idx_test, chan_train, chan_test, Label_train, Label_test, DC_train, DC_test) in enumerate(self.TrailKFold(data, chan_in_area, self.args.Label[sub], DC, n_splits=fold, shuffle=True)):   #生成训练集和测试集的索引
                content = '-------------------------------------------------------------'
                print(content)
                log2txt_pro(content,self.args.dataset_name)
                content = 'Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold)
                print(content)
                log2txt_pro(content,self.args.dataset_name)
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)

                acc_val, f1_val = self.first_stage(data=data_train, label=label_train, chan_in_area=chan_train, Label_train=Label_train,
                                                  subject=sub, fold=idx_fold, DC=DC_train)

                acc_test, pred, act = combine_train(args=self.args, data=data_train, data_test=data_test, label_test=label_test, label=label_train,
                              chan_train=chan_train, chan_test=chan_test, Label_train=Label_train, Label_test=Label_test,
                              subject=sub, fold=idx_fold, target_acc=1, DC_train=DC_train, DC_test=DC_test)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)
            sub_time = TIMER.measure()
            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts, classes=self.args.classes)
            tta.append(acc)
            ttf.append(f1)
            result = 'sub {} time:{}, eacc:{}, F1:{}'.format(sub, sub_time, tta[-1], f1)
            self.log2txt(result)
        tol_time = TIMER_.measure()
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final run_time: {}'.format(tol_time))
        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: test mean F1:{} std:{}'.format(mF1, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = 'Final test mAcc={} mF1={} val mAcc={} val mF1={}'.format(mACC,mF1, mACC_val, mF1_val)
        self.log2txt(results)


    def first_stage(self, data, label, chan_in_area, Label_train, subject, fold, DC):
        va = Averager()
        vf = Averager()
        va_item = []
        maxAcc = 0.0
        minLoss1 = 1000
        for i, (idx_train, idx_val, chan_train, chan_val, Label_train, Label_val, DC_train, DC_test) in enumerate(self.TrailKFold(data, chan_in_area, Label_train, DC, n_splits=self.args.n_splits1)):
            content = 'Inner 3-fold-CV Fold:{}'.format(i)
            print(content)
            log2txt_pro(content,self.args.dataset_name)
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            min_loss1, acc_val, F1_val = train(args=self.args,data_train=data_train,label_train=label_train, data_val=data_val,label_val=label_val,
            Label_train=Label_train, Label_val=Label_val, chan_train=chan_train, chan_val=chan_val, subject=subject, fold=fold, DC_train=DC_train, DC_test=DC_test)  #得到多个epoch中最好的正确率和F1分数
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            if min_loss1 <= minLoss1:
                minLoss1 = min_loss1
                old_ee_name = osp.join(self.args.save_path, 'candidate_ee.pth')
                new_ee_name = osp.join(self.args.save_path, 'min-ee-loss1.pth')
                if os.path.exists(new_ee_name):
                    os.remove(new_ee_name)
                os.rename(old_ee_name, new_ee_name)

            if acc_val >= maxAcc:
                maxAcc = acc_val

                old_name = osp.join(self.args.save_path, 'candidate.pth')
                new_name = osp.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                content = 'New max ACC model saved, with the val ACC being:{}'.format(acc_val)
                print(content)
                log2txt_pro(content,self.args.dataset_name)
        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()





