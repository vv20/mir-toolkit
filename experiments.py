import pandas as pd
import os
import unittest
import librosa
import mido
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import mir_eval
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from collections import Counter

import typesystem
import onset_detection
import music_input
import feature_extraction.pitch_features
import feature_extraction.melodic_features
import feature_extraction.dynamic_features
import feature_extraction.instrumentation_features

def averageOfMetric(evals, metric_name):
    return sum([e[metric_name] for e in evals]) / len(evals)

def printEvaluation(name, metric_names, evals):
    print(name)
    for metric_name in metric_names:
        print(metric_name + ": %.2f" % averageOfMetric(evals, metric_name))
    print()

class TestMidi(unittest.TestCase):
    def setUp(self):
        self.midi_dir_name = "../midi/ground-truth/"
        self.filename1 = self.midi_dir_name + "midi-Track_1-1.mid"
        self.filename2 = self.midi_dir_name + "midi-Track_2-1.mid"
        self.filename3 = self.midi_dir_name + "midi-Track_3-1.mid"
        self.filename4 = self.midi_dir_name + "ground_truth-export-1.mid"

    def sortByTimeAndPitch(self, events):
        events = [e for e in events if e.type == "note_on" or e.type == "note_off"]
        # convert delta time to absolute time
        current_time = 0
        for event in events:
            event.time = event.time + current_time
            current_time = event.time
        # sort
        events.sort(key=lambda e: e.time * 128 + e.note)
        # convert back to delta time
        current_time = 0
        for event in events:
            temp_time = current_time
            current_time = event.time
            event.time = event.time - temp_time
        return events

    def testSmoketest(self):
        midifile = mido.MidiFile(self.filename1)
        default = [msg for msg in list(midifile) 
                if msg.type == "note_on" or msg.type == "note_off"]
        default = self.sortByTimeAndPitch(default)
        result = music_input.from_midi(self.filename1)
        result = self.sortByTimeAndPitch(result.to_midi())
        self.assertListEqual(default, result)

    def testWithRests(self):
        midifile = mido.MidiFile(self.filename2)
        default = [msg for msg in list(midifile) 
                if msg.type == "note_on" or msg.type == "note_off"]
        result = music_input.from_midi(self.filename2)
        result = self.sortByTimeAndPitch(result.to_midi())
        self.assertListEqual(default, result)

    def testSimultaneousNotes(self):
        midifile = mido.MidiFile(self.filename3)
        default = [msg for msg in list(midifile) 
                if msg.type == "note_on" or msg.type == "note_off"]
        result = music_input.from_midi(self.filename3).to_midi()
        self.assertListEqual(default, result)

    def testDifferentVelocity(self):
        midifile = mido.MidiFile(self.filename4)
        default = [msg for msg in list(midifile) 
                if msg.type == "note_on" or msg.type == "note_off"]
        result = music_input.from_midi(self.filename4).to_midi()
        self.assertListEqual(default, result)


class TestTranscription(unittest.TestCase):
    def setUp(self):
        dataset_dir = "../datasets/midi/"
        self.sound_dir = dataset_dir + "sounds/"
        self.truth_dir = dataset_dir + "ground-truth/"
        self.out_dir = dataset_dir + "result/"
        self.out_dir2 = dataset_dir + "result2/"
        self.out_dir3 = dataset_dir + "result3/"
        self.soundfiles = os.listdir(self.sound_dir)
        self.metrics = [
                    "Precision",
                    "Recall",
                    "F-measure",
                    "Average_Overlap_Ratio",
                    "Precision_no_offset",
                    "Recall_no_offset",
                    "F-measure_no_offset",
                    "Average_Overlap_Ratio_no_offset",
                    "Onset_Precision",
                    "Onset_Recall",
                    "Onset_F-measure",
                    "Offset_Precision",
                    "Offset_Recall",
                    "Offset_F-measure"
                ]

    def testMelodia(self):
        # parse files into midi
        for f in self.soundfiles:
            resultname = self.out_dir
            for i in range(len(f.split(".")) - 1):
                resultname += f.split(".")[i]
                resultname += "."
            resultname += "mid"
            if os.path.isfile(resultname):
                continue
            print("parsing", f)
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            bpm = librosa.beat.tempo(signal, fs)
            subprocess.run(["python3", 
                "../audio_to_midi_melodia/audio_to_midi_melodia.py",
                        srcname, resultname, str(int(bpm[0]))])

        # evaluate
        evals = []
        truthfiles = os.listdir(self.truth_dir)
        for f in truthfiles:
            print("evaluating", f)
            truthname = self.truth_dir + f
            resultname = self.out_dir + f
            truths = music_input.from_midi(truthname).pitch_intervals()
            results = music_input.from_midi(resultname).pitch_intervals()
            truthintervals = np.asarray([[n["onset"],n["offset"]] for n in truths])
            truthpitches = np.asarray([n["pitch"] for n in truths])
            resultintervals = np.asarray([[n["onset"],
                n["offset"]] for n in results])
            resultpitches = np.asarray([n["pitch"] for n in results])
            evals.append(mir_eval.transcription.evaluate(truthintervals, 
                truthpitches, resultintervals, resultpitches))
        printEvaluation("Melodia Transcription", self.metrics, evals)

    def testPerFrameTranscription(self):
        # transcribe and pickle
        for s in self.soundfiles:
            print("transcribing", s)
            result = music_input.from_audio(self.sound_dir + s, 
                    algorithm="frame")
            resultname = self.out_dir2
            for i in range(len(s.split(".")) - 1):
                resultname += s.split(".")[i]
                resultname += "."
            resultname += "pickle"
            resultfile = open(resultname, "wb")
            pickle.dump(result, resultfile, protocol=pickle.HIGHEST_PROTOCOL)
            resultfile.close()

        # unpickle and evaluate
        evals = []
        resultfiles = os.listdir(self.out_dir2)
        for f in resultfiles:
            print("analysing " + f)
            midiname = self.truth_dir
            for i in range(len(f.split(".")) - 1):
                midiname += f.split(".")[i]
                midiname += "."
            midiname += "mid"
            resultfile = open(self.out_dir2 + f, "rb")
            result = pickle.load(resultfile).pitch_intervals()
            resultfile.close()
            truth = music_input.from_midi(midiname).pitch_intervals()
            truthintervals = np.asarray([[n["onset"],n["offset"]] for n in truth])
            truthpitches = np.asarray([n["pitch"] for n in truth])
            resultintervals = np.asarray([[n["onset"],
                n["offset"]] for n in result])
            resultpitches = np.asarray([n["pitch"] for n in result])
            evals.append(mir_eval.transcription.evaluate(truthintervals, 
                truthpitches, resultintervals, resultpitches))
        printEvaluation("Per-Frame Transcription", self.metrics, evals)

    def testExtractPitchesFromOnsets(self):
        # transcribe and pickle
        for s in self.soundfiles:
            print("transcribing", s)
            result = music_input.from_audio(self.sound_dir + s, 
                    algorithm="onset")
            resultname = self.out_dir3
            for i in range(len(s.split(".")) - 1):
                resultname += s.split(".")[i]
                resultname += "."
            resultname += "pickle"
            resultfile = open(resultname, "wb")
            pickle.dump(result, resultfile, protocol=pickle.HIGHEST_PROTOCOL)
            resultfile.close()

        # unpickle and evaluate
        evals = []
        resultfiles = os.listdir(self.out_dir3)
        for f in resultfiles:
            print("analysing " + f)
            midiname = self.truth_dir
            for i in range(len(f.split(".")) - 1):
                midiname += f.split(".")[i]
                midiname += "."
            midiname += "mid"
            resultfile = open(self.out_dir3 + f, "rb")
            result = pickle.load(resultfile).pitch_intervals()
            resultfile.close()
            truth = music_input.from_midi(midiname).pitch_intervals()
            truthintervals = np.asarray([[n["onset"],n["offset"]] for n in truth])
            truthpitches = np.asarray([n["pitch"] for n in truth])
            resultintervals = np.asarray([[n["onset"],
                n["offset"]] for n in result])
            resultpitches = np.asarray([n["pitch"] for n in result])
            evals.append(mir_eval.transcription.evaluate(truthintervals, 
                truthpitches, resultintervals, resultpitches))
        printEvaluation("From-Onset Transcription", self.metrics, evals)


class TestDetectTempo(unittest.TestCase):
    def setUp(self):
        dataset_dir = "../datasets/giantsteps-tempo-dataset/"
        self.sound_dir = dataset_dir + "audio/"
        self.truth_dir = dataset_dir + "annotations/tempo/"
        self.soundfiles = os.listdir(self.sound_dir)
        self.metrics = ["P-score"]

    def testLibrosaTempo(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir
            for i in range(len(f.split(".")) - 1):
                truthname += f.split(".")[i]
                truthname += "."
            truthname += "bpm"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = librosa.beat.tempo(signal, fs)
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.tempo.evaluate(
                reference_tempi = np.asarray([truth / 2, truth]),
                reference_weight = 0.5,
                estimated_tempi = np.asarray([result / 2, result])))
            print(evals[-1])
        printEvaluation("Librosa Tempo Detection", self.metrics, evals)


class TestDetectOnsets(unittest.TestCase):
    def setUp(self):
        dataset_dir = "../datasets/ODB/"
        self.sound_dir = dataset_dir + "sounds/"
        self.truth_dir = dataset_dir + "ground-truth/"
        self.soundfiles = os.listdir(self.sound_dir)
        self.metrics = [
                "F-measure",
                "Precision",
                "Recall"
                ]

    def testCalculateBelloHFCEval(self):
        truth = np.asarray([93, 489, 212, 271])
        false_positives = np.asarray([0.147, 0.054, 0, 0.0108])
        true_positives = np.asarray([0.817, 0.941, 0.967, 0.845])

        false_positives = false_positives * truth
        true_positives = true_positives * truth
        positives = false_positives + true_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testCalculateBelloSDEval(self):
        truth = np.asarray([93, 489, 212, 271])
        false_positives = np.asarray([0.086, 0.016, 0.055, 0.104])
        true_positives = np.asarray([0.871, 0.949, 0.816, 0.804])

        false_positives = false_positives * truth
        true_positives = true_positives * truth
        positives = false_positives + true_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testCalculateBelloPDEval(self):
        truth = np.asarray([93, 489, 212, 271])
        false_positives = np.asarray([0.043, 0.003, 0.055, 0.247])
        true_positives = np.asarray([0.957, 0.955, 0.807, 0.801])

        false_positives = false_positives * truth
        true_positives = true_positives * truth
        positives = false_positives + true_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testCalculateBelloWRMEval(self):
        truth = np.asarray([93, 489, 212, 271])
        false_positives = np.asarray([0.101, 0.051, 0.022, 0.277])
        true_positives = np.asarray([0.925, 0.927, 0.887, 0.819])

        false_positives = false_positives * truth
        true_positives = true_positives * truth
        positives = false_positives + true_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testCalculateBelloNLLEval(self):
        truth = np.asarray([93, 489, 212, 271])
        false_positives = np.asarray([0.032, 0.031, 0.017, 0.108])
        true_positives = np.asarray([0.968, 0.924, 0.929, 0.86])

        false_positives = false_positives * truth
        true_positives = true_positives * truth
        positives = false_positives + true_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testCalculateKlapuriEval(self):
        truth = np.asarray([59, 62, 49, 19, 51, 34, 46, 51, 33, 30])
        false_negatives = np.asarray([3, 5, 4, 1, 3, 2, 5, 3, 7, 0])
        false_positives = np.asarray([0, 1, 1, 2, 0, 1, 1, 1, 10, 28])

        true_positives = truth - false_negatives
        positives = true_positives + false_positives

        precision = true_positives / positives
        recall = true_positives / truth
        f_measure = 2 * (precision * recall) / (precision + recall)

        print("Precision:", np.mean(precision))
        print("Recall:", np.mean(recall))
        print("F-measure:", np.mean(f_measure))

    def testLibrosaOnsets(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = librosa.onset.onset_detect(signal, fs, units="time")
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("Librosa onset detection", self.metrics, evals)

    def testHFC(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="hfc", 
                    units="time")
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("HFC onset detection", self.metrics, evals)

    def testHFCMultiband(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="hfc", 
                    units="time", split_bands=True)
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("HFC onset detection with multiple bands", self.metrics, 
                evals)

    def testHFCOptimalThreshold(self):
        thresholds = map(lambda x: x / 20, range(0, 40, 5))
        best_evals = None
        best_f = 0
        best_threshold = 0
        for threshold in thresholds:
            evals = []
            for f in self.soundfiles:
                truthname = self.truth_dir + f.split(".")[0] + ".txt"
                srcname = self.sound_dir + f
                signal, fs = librosa.load(srcname)
                result = onset_detection.detect_onsets(signal, fs, rt="hfc", 
                        units="time", split_bands=True, threshold=threshold)
                truth = mir_eval.io.load_events(truthname)
                evals.append(mir_eval.onset.evaluate(truth, result))
            f = averageOfMetric(evals, "F-measure")
            if f > best_f:
                best_f = f
                best_threshold = threshold
                best_evals = evals
        print("Optimal threshold", best_threshold)
        printEvaluation("HFC optimal threshold", self.metrics, best_evals)

    def testSD(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="sd", units="time")
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("SD onset detection", self.metrics, evals)

    def testSDMultiband(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="sd",
                    units="time", split_bands=True)
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("SD onset detection with multiple bands", self.metrics, 
                evals)

    def testPD(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="pd", 
                    units="time")
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("PD onset detection", self.metrics, evals)

    def testPDMultiband(self):
        evals = []
        for f in self.soundfiles:
            truthname = self.truth_dir + f.split(".")[0] + ".txt"
            srcname = self.sound_dir + f
            signal, fs = librosa.load(srcname)
            result = onset_detection.detect_onsets(signal, fs, rt="pd",
                    units="time", split_bands=True)
            truth = mir_eval.io.load_events(truthname)
            evals.append(mir_eval.onset.evaluate(truth, result))
        printEvaluation("PD onset detection with multiple bands", self.metrics, 
                evals)

class TestGenreClassification(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = "../datasets/genre-midi/ground-truth/"
        self.genres = os.listdir(self.dataset_dir)
        self.feature_file = "features.csv"
        self.one_dimensional_file = "one-dimensional.csv"
        self.multi_dimensional_file = "multi-dimensional.csv"

    def extractOneDimensionalFeatures(self):
        if os.path.isfile(self.one_dimensional_file):
            return pd.read_csv(self.one_dimensional_file, sep=",")
        names = []
        features = []
        for i in range(len(self.genres)):
            files = os.listdir(self.dataset_dir + self.genres[i])
            for f in files:
                m = music_input.from_midi(self.dataset_dir + self.genres[i] + "/" + f)
                pitch_feat = feature_extraction.pitch_features.one_dimensional_features(m)
                inst_feat = feature_extraction.instrumentation_features.one_dimensional_features(m)
                dyn_feat = feature_extraction.dynamic_features.one_dimensional_features(m)
                mel_feat = feature_extraction.melodic_features.one_dimensional_features(m)

                this_names = pitch_feat[0]
                this_names.extend(inst_feat[0])
                this_names.extend(dyn_feat[0])
                this_names.extend(mel_feat[0])
                if len(names) == 0:
                    names = this_names
                elif len(names) != len(this_names):
                    raise TypeError

                feat_vector = pitch_feat[1]
                feat_vector.extend(inst_feat[1])
                feat_vector.extend(dyn_feat[1])
                feat_vector.extend(mel_feat[1])
                feat_vector.append(i)
                features.append(feat_vector)
        names.append("genre")
        df = pd.DataFrame(data=np.asarray(features), columns=names)
        df.to_csv(self.one_dimensional_file, sep=",")

    def extractMultiDimensionalFeatures(self):
        if os.path.isfile(self.multi_dimensional_file):
            return
        names = []
        features = []
        for i in range(len(self.genres)):
            files = os.listdir(self.dataset_dir + self.genres[i])
            for f in files:
                m = music_input.from_midi(self.dataset_dir + self.genres[i] + "/" + f)
                pitch_feat = feature_extraction.pitch_features.multi_dimensional_features(m)
                inst_feat = feature_extraction.instrumentation_features.multi_dimensional_features(m)
                dyn_feat = feature_extraction.dynamic_features.multi_dimensional_features(m)
                mel_feat = feature_extraction.melodic_features.multi_dimensional_features(m)

                this_names = pitch_feat[0]
                this_names.extend(inst_feat[0])
                this_names.extend(dyn_feat[0])
                this_names.extend(mel_feat[0])
                if len(names) == 0:
                    names = this_names
                elif len(names) != len(this_names):
                    raise TypeError

                feat_vector = pitch_feat[1]
                feat_vector.extend(inst_feat[1])
                feat_vector.extend(dyn_feat[1])
                feat_vector.extend(mel_feat[1])
                feat_vector.append(i)
                features.append(feat_vector)
        names.append("genre")
        df = pd.DataFrame(data=np.asarray(features), columns=names)
        df.to_csv(self.multi_dimensional_file, sep=",")

    def extractFeatures(self):
        if os.path.isfile(self.feature_file):
            return
        one_dim = self.extractOneDimensionalFeatures()
        del one_dim["genre"]
        multi_dim = self.extractMultiDimensionalFeatures()
        all_features = pd.concat([one_dim, multi_dim], axis=1)
        all_features.to_csv(self.feature_file, sep=",")

    def printConfusionMatrix(self, y_test, y_predict, title):
        cm = confusion_matrix(y_test, y_predict)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks = np.arange(cm.shape[1]),
                yticks = np.arange(cm.shape[0]),
                xticklabels = self.genres,
                yticklabels = self.genres,
                title = title,
                ylabel = "True Genre",
                xlabel = "Predicted Genre")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i,j] > cm.max() / 2:
                    colour = "white"
                else:
                    colour = "black"
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=colour)
        fig.tight_layout()
        plt.show()

    def testRFC(self):
        self.extractOneDimensionalFeatures()
        df = pd.read_csv(self.one_dimensional_file)
        table = df.values
        names = df.columns[1:]
        x = df.values[1:,:-1]
        y = df.values[1:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)

        y_predict = rfc.predict(x_test)
        self.printConfusionMatrix(y_test, y_predict, "Random Forest Classifier")
        print(classification_report(y_test, y_predict))

        importances = []
        f = open("importances.csv", "w")
        for i in range(len(rfc.feature_importances_)):
            importances.append((i, rfc.feature_importances_[i]))
        importances = sorted(importances, key = lambda i: i[1], reverse = True)
        for i in range(10):
            index = importances[i][0]
            print(names[index], ",", rfc.feature_importances_[index], file=f)
        f.close()

        category_weights = {
                "pitch": 0.0,
                "melodic": 0.0,
                "dynamic": 0.0,
                "instrumentation": 0.0
                }
        pitch_names = feature_extraction.pitch_features.feature_names()
        melodic_names = feature_extraction.melodic_features.feature_names()
        dynamic_names = feature_extraction.dynamic_features.feature_names()
        instr_names=  feature_extraction.instrumentation_features.feature_names()
        for i in range(len(names)):
            if names[i] in pitch_names:
                category_weights["pitch"] += rfc.feature_importances_[i]
            if names[i] in melodic_names:
                category_weights["melodic"] += rfc.feature_importances_[i]
            if names[i] in dynamic_names:
                category_weights["dynamic"] += rfc.feature_importances_[i]
            if names[i] in instr_names:
                category_weights["instrumentation"] += rfc.feature_importances_[i]
        f = open("category-weights.csv", "w")
        for category in category_weights.keys():
            print(category, ",", category_weights[category], file=f)
        f.close()

    def testKNN(self):
        df = self.extractFeatures()
        table = df.values
        x = df.values[:,:-1]
        y = df.values[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)

        y_predict = knn.predict(x_test)
        self.printConfusionMatrix(y_test, y_predict, 
                "K-Nearest Neighbours Classifier")
        print(classification_report(y_test, y_predict))

    def testSVC(self):
        df = self.extractFeatures()
        df = pd.read_csv(self.feature_file, sep=",")
        table = df.values
        x = df.values[:,:-1]
        y = df.values[:,-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

        svc = LinearSVC()
        svc.fit(x_train, y_train)

        y_predict = svc.predict(x_test)
        self.printConfusionMatrix(y_test, y_predict, 
                "Support Vector Classifier")
        print(classification_report(y_test, y_predict))

    def testEnsemble(self):
        self.extractOneDimensionalFeatures()
        self.extractMultiDimensionalFeatures()
        one_dim = pd.read_csv(self.one_dimensional_file)
        multi_dim = pd.read_csv(self.multi_dimensional_file)
        one_dim_x = one_dim.values[:,:-1]
        multi_dim_x = multi_dim.values[:,:-1]
        one_dim_feat_no = one_dim_x.shape[1]
        y = one_dim.values[:,-1]
        x = np.concatenate([one_dim_x, multi_dim_x], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        one_dim_train_x = x_train[:,:one_dim_feat_no]
        one_dim_test_x = x_test[:,:one_dim_feat_no]
        multi_dim_train_x = x_train[:,one_dim_feat_no:]
        multi_dim_test_x = x_test[:,one_dim_feat_no:]

        ensemble = VotingEnsemble(KNeighborsClassifier, MLPClassifier)
        ensemble.fit(one_dim_train_x, multi_dim_train_x, y_train)
        y_predict = ensemble.predict(one_dim_test_x, multi_dim_test_x)
        self.printConfusionMatrix(y_test, y_predict, "Ensemble Classifier")
        print(classification_report(y_test, y_predict))

class VotingEnsemble:
    def __init__(self, classifier_ctor1, classifier_ctor2, no_classifiers=10):
        self.classifier1_instances = []
        for i in range(no_classifiers):
            self.classifier1_instances.append(classifier_ctor1())
        self.classifier2_instances = []
        for i in range(no_classifiers):
            self.classifier2_instances.append(classifier_ctor2())

    def fit(self, x1, x2, y):
        for c in self.classifier1_instances:
            c.fit(x1, y)
        for c in self.classifier2_instances:
            c.fit(x2, y)

    def predict(self, x1, x2):
        labels = []
        for c in self.classifier1_instances:
            labels.append(c.predict(x1))
        for c in self.classifier2_instances:
            labels.append(c.predict(x2))
        labels = np.asarray(labels)
        y = []
        for i in range(labels.shape[1]):
            y.append(Counter(labels[:,i]).most_common(1)[0][0])
        return np.asarray(y)

if __name__ == "__main__":
    unittest.main()
