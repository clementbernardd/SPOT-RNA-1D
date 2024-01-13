import tensorflow as tf
import numpy as np
from typing import Dict, List
import os
import tqdm
import json
norm_mu = [0.24416392, 0.19836862, 0.30642843, 0.24948534]
norm_std = [0.43031312, 0.39954973, 0.46168336, 0.43343267]
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class HelperCLI:
    def __init__(self, gpu: bool = False):
        self.gpu = 0 if gpu else None
        self.cpu = 6
        self.batch_size = 10
        self.meta_path = os.path.join("checkpoints/tensorflow_model.meta")
        self.checkpoint_path = os.path.join("checkpoints/tensorflow_model")

    def get_config(self):
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=self.cpu,
                                              inter_op_parallelism_threads=self.cpu)
        else:
            config = tf.compat.v1.ConfigProto()
            config.allow_soft_placement = True
            config.log_device_placement = False
        return config

    def predict(self, sequence: str, ids: str):
        """
        Predict the torsional angles from a given sequence using SPOT-RNA-1D
        """
        sequence = self._clean_sequence(sequence)
        try:
            feat_dic = self.create_one_hot_encoding([ids], [sequence])
        except ValueError:
            return {}
        outputs = self.tf_pred(feat_dic, ids)
        output_json = self.convert_preds_to_json(outputs, [ids], [sequence])
        return output_json

    def convert_output_to_json(self, final_output: np.ndarray):
        sequence = ''.join(final_output[:, 1])
        columns = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi', 'eta', 'theta']
        output = {'sequence': sequence, 'angles': {}}
        for i in range(2, final_output.shape[1]):
            angles = list(final_output[:, i])
            angles = [float(i) for i in angles]
            output['angles'][columns[i - 2]] = angles
        return output

    def convert_preds_to_json(self, outputs: np.ndarray, ids: List, sequences: List):
        json_output = {}
        for index, c_id in enumerate(ids):
            seq = np.array([[i, I] for i, I in enumerate(sequences[index])])
            preds = outputs[c_id] * 2 - 1
            preds_angle_rad = [np.arctan2(preds[:, 2 * i], preds[:, 2 * i + 1]) for i in
                               range(int(preds.shape[1] / 2))]
            preds_angle = np.round(np.transpose(np.degrees(preds_angle_rad)), 2)
            final_output = np.concatenate((seq, preds_angle), axis=1)
            output = self.convert_output_to_json(final_output)
            json_output[c_id] = output
        return json_output

    def tf_pred(self, feat_dic: Dict, ids):
        outputs = {}
        config = self.get_config()
        count = 1
        with tf.compat.v1.Session(config=config) as sess:
            saver = tf.compat.v1.train.import_meta_graph(self.meta_path)
            saver.restore(sess, self.checkpoint_path)
            graph = tf.compat.v1.get_default_graph()
            tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')
            for batch_test in tqdm.tqdm(range(max(1, int(np.ceil(count / self.batch_size))))):
                feature, mask, seq_lens, batch_ids = self.get_data(feat_dic, [ids],
                                                              batch_size=self.batch_size,
                                                              i=batch_test, norm_mu=norm_mu,
                                                              norm_std=norm_std)
                out = sess.run([tmp_out],
                               feed_dict={'input_feature:0': feature, 'seq_lens:0': seq_lens,
                                          'zero_mask:0': mask, 'keep_prob:0': 1.})
                pred_angles = self.sigmoid(out[0])
                for i, c_id in enumerate(batch_ids):
                    outputs[c_id] = pred_angles[i, 0:seq_lens[i]]
        tf.compat.v1.reset_default_graph()
        return outputs

    def create_one_hot_encoding(self, ids, sequences):
        feat_dic = {}
        bases = np.array([base for base in 'AUGC'])
        for i,I in enumerate(ids):
            feat_dic[I] = np.concatenate([[(bases==base.upper()).astype(int)] if str(base).upper() in 'AUGC' else np.array([[0]*4]) for base in sequences[i]])  # one-hot encoding Lx4
        return feat_dic
        
    def _clean_sequence(self, sequence: str) -> str:
        return sequence.replace(" ","").upper().replace("T", "U")

    def get_data(self, sample_feat, ids, batch_size, i, norm_mu,norm_std):
        ###############################################
        # prepare normalize input feature
        # make batch of input features
        # prepare zero-mask for different length input sequences
        ###############################################
        data = [(sample_feat[j][:,:]-norm_mu)/norm_std for j in ids[i * batch_size:np.min([(i + 1) * batch_size, len(ids)])]]
        data = [np.concatenate([np.ones((j.shape[0], 1)), j], axis=1) for j in data]
        seq_lens = [j.shape[0] for j in data]
        batch_ids = [j for j in ids[i * batch_size:np.min([(i + 1) * batch_size, len(ids)])]]
        max_seq_len = max(seq_lens)
        data = np.concatenate([np.concatenate([j, np.zeros((max_seq_len - j.shape[0], j.shape[1]))])[None, :, :] for j in data])
        mask = np.concatenate([np.concatenate([np.ones((1, seq_lens[j])), np.zeros((1, max_seq_len - seq_lens[j]))], axis=1) for j in range(len(ids[i * batch_size:np.min(((i + 1) * batch_size, len(ids)))]))])
        return data, mask, seq_lens, batch_ids

    def sigmoid(self, x):
        return x#1 / (1 + np.exp(-x))

    def predict_from_dir_json(self, in_path: str, out_path: str):
        """
        Predict the torsional angles from a given sequence using SPOT-RNA-1D
        """
        outputs = {}
        if os.path.isfile(in_path) and in_path.endswith(".json"):
            content = self.read_json(in_path)
            ids = list(content.keys())
            sequences = [self._clean_sequence(content[i]["sequence"]) for i in ids]
            for sequence, c_id in zip(sequences, ids):
                output = self.predict(sequence, c_id)
                outputs = {**outputs, **output}
        self.save_json(outputs, out_path)

    def read_json(self, path: str):
        """Read a json file and return the content as a dictionary.
        Args
        :param path: the path to the json file
        :return: the content of the json file as a dictionary
        """
        with open(path, "r") as file:
            content = json.load(file)
        return content


    def save_json(self, content, path: str):
        """Save the dictionary into a json file.
        Args
        :param content: the object to save
        :param path: the path where to save. Could have .json or not in the path
        """
        assert(type(content) is dict)
        if path.endswith(".json"):
            path_to_save = path
        else:
            path_to_save = path + ".json"
        with open(path_to_save, "w") as file:
            json.dump(content, file)

if __name__ == "__main__":
    in_paths = [
        os.path.join("data", name) for name in  [
            "all_pdb.json", "casp_rna.json", "rna_puzzles.json", "train.json", "valid.json"
        ]
    ]
    out_paths = [
        os.path.join("outputs", os.path.basename(name)) for name in in_paths
    ]
    for in_path, out_path in zip(in_paths, out_paths):
        helper_cli = HelperCLI(gpu=False)
        helper_cli.predict_from_dir_json(in_path, out_path)
        print(f"Predicting {in_path} and saving the results in {out_path}")
