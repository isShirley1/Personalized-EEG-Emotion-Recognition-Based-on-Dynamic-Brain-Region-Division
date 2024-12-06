import _pickle as cPickle
from train_model import *
from scipy import signal
from sklearn import preprocessing
import scipy.io as scio


class PrepareData:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.sfreq = args.sfreq


    def load_data_per_subject(self, sub):
        sub += 1
        if self.dataset_name == 'DEAP':
            if (sub < 10):
                sub_code = str('s0' + str(sub) + '.dat')
            else:
                sub_code = str('s' + str(sub) + '.dat')
            path = r'D:\DEAP\data_preprocessed_python'
            subject_path = os.path.join(path, sub_code)
            subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
            label = subject['labels']
            Xbase = subject['data'][:, 0:32, :3 * 128]
            Xbase_ = np.mean(Xbase, axis=2)
            data_ = subject['data'][:, 0:32, 3 * 128:]
            data = data_- Xbase_[:,:,np.newaxis]
            label = self.label_switching(label)
            print('subject{} binary label generated!'.format(sub))
        else:
            if self.dataset_name == 'SEED':
                path = r'D:\SEED\Preprocessed_EEG'
                path_label = os.path.join(path, r'label.mat')
                label = np.tile(scio.loadmat(path_label)['label'][0]+1, 3)
            else:
                path = r'D:\SEED_IV\Preprocessed_EEG'
                path_label = os.path.join(path, r'label.mat')
                label = np.hstack((scio.loadmat(path_label)['label'][0],scio.loadmat(path_label)['label'][1],scio.loadmat(path_label)['label'][2]))

            data = []
            for i in range(3):
                path_feature = os.path.join(path,'session'+str(i+1),str(sub)+'.mat')
                temp0 = scio.loadmat(path_feature)
                a = 1
                for j in temp0.keys():
                    if a > 3:
                        data.append(temp0[j])
                    a=a+1
        return data, label

    def DE_extract(self,data , fs):
        fStart = [0.5, 4, 8, 13, 32]
        fEnd = [4, 8, 13, 32, 50]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        Bands_data = []
        padding_length = 238 if self.dataset_name == 'SEED' else 216 if self.dataset_name == 'SEED-IV' else False
        for i in range(data.shape[0]):
            feature = min_max_scaler.fit_transform(data[i]).astype('float32')
            band_data = []
            for band_index, band in enumerate(fStart):
                b, a = signal.butter(4, [fStart[band_index] / fs, fEnd[band_index] / fs],
                                     'bandpass')
                filtedFeature = signal.filtfilt(b, a, feature)
                filtedFeature_de = np.empty((self.args.channel_num,0))
                for de_index in range(0, filtedFeature.shape[1], fs):
                    variance = np.var(filtedFeature[:, de_index: de_index + fs], axis=1, ddof=1).reshape(-1, 1)  # 求得方差
                    DE = np.log(2 * math.pi * math.e * variance) / 2
                    filtedFeature_de = np.hstack([filtedFeature_de, DE])
                band_data.append(filtedFeature_de)
            Bands_data.append(np.array(band_data).transpose((1,2,0)))
        return np.array(Bands_data)

    def label_switching(self, label):
        if self.args.label_type == 'A':
            label = label[:, 1]
        elif self.args.label_type == 'V':
            label = label[:, 0]
        label_swiched = np.where(label <= 5, 0, 1)
        return label_swiched

    def run(self, subject_list, split):
        Label = []
        for sub in subject_list:
            data, label = self.load_data_per_subject(sub)
            Label.append(label)
            if split:
                data_, label_ = self.split(data=data, label=label, segment_length=self.args.segment_length,
                                            overlap=self.args.overlap,sampling_rate=self.sfreq)
            self.save(data_, label_, sub, False)
            data = self.DE_extract(data_, self.sfreq)
            print('subject{} Data and label prepared!'.format(sub+1))
            print('---------------------------')
            self.save(data, label_, sub,True)
        num_segment = int(label_.shape[0]/self.args.trails)
        save_path = os.getcwd()
        path_name = 'data_{}_{}'.format(self.dataset_name, 'extracted')
        save_path = os.path.join(save_path, path_name, 'other_parameters.mat')
        varialbe_name = {'num_segment': num_segment, 'Label':Label}
        scio.savemat(save_path, varialbe_name)


    def split(self, data, label, overlap = 0,segment_length=1, sampling_rate=128):
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []
        if self.dataset_name == 'DEAP':
            data_point = data.shape[-1]
            number_segment = int((data_point - data_segment) // step) + 1
            for i in range(number_segment):
                data_split.append(data[:, :, (i * step):(i * step + data_segment)])
            data_split_array = np.stack(data_split, axis=1)

        else:
            cutpoint = 232*200 if self.dataset_name == 'SEED' else 165*200
            number_segment = int((cutpoint - data_segment) // step) +1
            for trail in range(len(data)):
                data_trail = np.array(data[trail])
                data_point = data_trail.shape[1]
                trail_split = []
                if data_point >= cutpoint:
                    for i in range(number_segment):
                        trail_split.append(data_trail[:, (i * step):(i * step + data_segment)])
                    trail_split_array = np.stack(trail_split, axis=0)
                else:
                    segment = int((data_point - data_segment) // step)
                    for i in range(segment):
                        trail_split.append(data_trail[:, (i * step):(i * step + data_segment)])
                    trail_split_array = np.stack(trail_split, axis=0)
                    padarray = np.mean(trail_split_array, axis=0)
                    for j in range(number_segment-segment):
                        trail_split_array = np.concatenate((trail_split_array, [padarray]), axis=0)
                data_split.append(trail_split_array)
            data_split_array = np.stack(data_split,axis=0)
        data = data_split_array.reshape((-1, data_split_array.shape[2], data_split_array.shape[3]))
        label = np.hstack([np.repeat(label[i], int(number_segment)) for i in range(len(label))])

        assert len(data) == len(label)
        return data, label

    def save(self, data, label, sub, data_type):
        save_path = os.getcwd()
        path_name = 'data_{}_{}'.format( self.dataset_name, 'extracted' if data_type else 'origion' )
        save_path = os.path.join(save_path, path_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub+1) + '.mat'
        save_path = os.path.join(save_path, name)
        varialbe_name = {'data':data,'label':label}
        scio.savemat(save_path,varialbe_name)

