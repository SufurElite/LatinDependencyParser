import os
from sklearn.model_selection import train_test_split
from conllu import parse
import random
from random import randint
from sentence_transformers import InputExample
import json
import pandas as pd

DATA_DIR = "../data/"
OUTPUT_DATA = "../combined_data/"

"""
The following dictionary was created by going over the latin treebank repo
for Perseus DL, which can be found here: 
    https://github.com/PerseusDL/treebank_data/blob/master/v2.1/Latin/texts/
The point is that the document ids here correspond to the CONLLU doc ids, so we can
determine the author. The list of authors is also on the UniversalDependencies Page: 
    https://github.com/UniversalDependencies/UD_Latin-Perseus
"""
perseus_id_to_author = {
    "phi0448.phi001": "Caesar",  # Gaius Iulius Caesar
    "phi0474.phi013": "Cicero",  # Marcus Tullius Cicero
    "phi0620.phi001": "Propertius",  # Propertius
    "phi0631.phi001": "Sallust",  # C. Sallusti Crispi
    "phi0690.phi003": "Vergil",  # Publius Vergilius Maro
    "phi0959.phi006": "Ovid",  # Ovid
    "phi0972.phi001": "Petronius",  # Petronius Arbiter
    "phi0975.phi001": "Phaerus",  # Phaedrus, Augusti libertus
    "phi1221.phi007": "Augustus",  # Res Gestae, Augustus? - author not listed in the github repo
    "phi1348.abo012": "Suetonius",  # C. Suetonius Tranquillus
    "phi1351.phi005": "Tacitus",  # Cornelius Tacitus
    "tlg0031.tlg027": "Jerome",  # Jerome, Vulgate Bible
}


"""
For PROIEL,
The treebank contains:
    * most of the Vulgate New Testament translation 
    * selections from Caesar's Gallic War 
    * Cicero's Letters to Atticus. 
Thus a similar text to author dictionary could be created - this maps
the first word of the source metadata to its author
"""

proiel_to_author = {
    "Jerome's": "Jerome",
    "De": "Cicero",
    "Epistulae": "Cicero",
    "Opus": "Palladius",
    "Commentarii": "Caesar",
}


author_to_age = {
    "Dante": "Medieval",
    "Jerome": "Classical",  # Described as classical Latin
    "Palladius": "Late",  # Late or Classical?
    "Cicero": "Classical",
    "Caesar": "Classical",
    "Aquinas": "Medieval",
    "Late": "Late",
    "Phaerus": "Classical",
    "Augustus": "Classical",
    "Suetonius": "Classical",
    "Tacitus": "Classical",
    "Propertius": "Classical",
    "Sallust": "Classical",
    "Vergil": "Classical",
    "Ovid": "Classical",
    "Petronius": "Classical",
    "test": "test",
}


def print_proportions(ages_to_nums):
    total = 0.0
    for key in ages_to_nums:
        total += ages_to_nums[key]
    for key in ages_to_nums:
        print(key, ages_to_nums[key] / total)


def load_data(
    data_dir: str = DATA_DIR, train_for_sbert: bool = False, return_sents: bool = False
):
    text_sents = set()
    total_data = {"sent": [], "author": [], "age": [], "conllus": []}
    total_available = 0

    print("{:30s} | {:11s} | {:6s}".format("Filename", "Amount used", "Amount Skipped"))
    print("=" * 61)
    author_to_sents = {}
    latin_age_to_num_sents = {}
    sbert_sents = []

    for path, dirs, filenames in os.walk(data_dir):
        dirs.sort()
        for filename in filenames:
            if filename.endswith("conllu"):
                file_path = os.path.join(path, filename)
                with open(file_path, "r") as f:
                    data = parse(f.read())
                pre_check = len(data)
                skip = 0
                for sent in data:
                    sent_text = sent.metadata["text"]
                    if sent_text in text_sents:
                        skip += 1
                        continue
                    text_sents.add(sent_text)
                    total_data["conllus"].append(sent)
                    author = ""

                    dirpath = os.path.dirname(file_path)
                    if dirpath.endswith("Perseus"):
                        sent_id = sent.metadata["sent_id"]
                        first_period = sent_id.find(".") + 1
                        second_period = sent_id[first_period:].find(".")
                        doc_id = sent_id[: first_period + second_period]
                        author = perseus_id_to_author[doc_id]
                    elif dirpath.endswith("PROIEL"):
                        source = sent.metadata["source"]
                        first_space = source.find(" ")
                        first_word = source[:first_space]
                        author = proiel_to_author[first_word]
                    elif dirpath.endswith("ITTB"):
                        author = "Aquinas"  # not strictly true, some are misattributed to him
                    elif dirpath.endswith("Dante"):
                        author = "Dante"
                    elif dirpath.endswith("Late"):
                        author = "Late"
                    elif dirpath.endswith("test_data") or dirpath.endswith(
                        "combined_data"
                    ):
                        author = "test"
                    else:
                        author = "error"

                    age = author_to_age[author]
                    if age not in latin_age_to_num_sents:
                        latin_age_to_num_sents[age] = 0

                    if author not in author_to_sents:
                        author_to_sents[author] = []

                    author_to_sents[author].append(sent_text)
                    latin_age_to_num_sents[age] += 1

                    if train_for_sbert:
                        sbert_sents.append([sent_text, author])
                    elif return_sents:
                        total_data["sent"].append(sent_text)
                        total_data["author"].append(author)
                        total_data["age"].append(age)

                print(f"{filename:30s} : {pre_check:11.1f} : {skip:14.1f}")
                total_available += pre_check
    print("=" * 61)
    print(
        f"Total unique data : {len(text_sents)} | filtered out {total_available - len(text_sents)}"
    )

    print("=" * 61)
    print("\n")

    print("The Latin Age sentence distribution is as follows: ")
    print(json.dumps(latin_age_to_num_sents, indent=2))
    print("as a percentage:")
    print_proportions(latin_age_to_num_sents)

    if train_for_sbert:
        return sbert_sents
    elif return_sents:
        return pd.DataFrame(total_data)

    return author_to_sents


def prepare_data_for_sbert_training(
    data_path: str = DATA_DIR, training_size: int = 50000
):
    """
    Given the data from load_data, we want to be able to create
    sentence representations, which ideally are different according to the
    stage at which the Latin was written.
    According to the sbert page on training data
        https://www.sbert.net/docs/training/overview.html#training-data
    they suggest that labeling is dependent on the task at hand and they suggest
    for instance, sentences of the same document being similar or of neighboring sentences.

    Along the same lines, text from the same author will be labeled 1.0 and text from the same age
    but not the same author will be labelled .8, text from other ages will be labeled as 0.0 .
    This is a rudimentary approach, but is intended to just see what happens. I'll use a training split of the data
    to do this and then cluster with the test to see how well it performs at distinguishing by age.
    """

    sbert_sents = load_data(data_path, True)
    random.Random(42).shuffle(sbert_sents)
    train_dist = {}

    training_examples = []

    pairings = set()

    for _ in range(training_size):
        first_idx = randint(0, len(sbert_sents) - 1)
        second_idx = randint(0, len(sbert_sents) - 1)

        pair = (first_idx, second_idx)

        if (first_idx == second_idx) or (pair in pairings):
            continue
        first_author = sbert_sents[first_idx][1]
        second_author = sbert_sents[second_idx][1]

        first_age = author_to_age[first_author]
        second_age = author_to_age[second_author]

        pairings.add(pair)
        label_score = 0.0
        if first_author == second_author:
            label_score = 1.0
        elif first_age == second_age:
            label_score = 0.8

        training_examples.append(
            InputExample(
                texts=[
                    sbert_sents[first_idx][0],
                    sbert_sents[second_idx][0],
                ],
                label=label_score,
            )
        )

        combo = (first_age, second_age)
        reverse_combo = (second_age, first_age)
        if combo not in train_dist and reverse_combo in train_dist:
            combo = reverse_combo
        elif combo not in train_dist:
            train_dist[combo] = 0
        train_dist[combo] += 1

    print("The Latin Age combo sentence distribution is as follows: ")
    print_proportions(train_dist)

    return training_examples


def combine_data():
    def category_percentage(age_labels, values):
        total = 0.0
        for key in age_labels:
            total += values.count(key)
        for key in age_labels:
            print(key, values.count(key) / total)

    def write_conllu_file(location, file_name, data):
        output = ""
        for conllu_value in data:
            output += conllu_value.serialize()
        with open(os.path.join(location, f"{file_name}.conllu"), "w+") as f:
            f.write(output)

    total_data = load_data(DATA_DIR, return_sents=True)
    total_data.age = total_data.age.astype("category")
    X = total_data.conllus
    y = total_data.age
    print("starting y")
    X_train, X_remaining, y_train, remaining_y = train_test_split(X, y, test_size=0.33)
    write_conllu_file(OUTPUT_DATA, "train", X_train)
    print("y_train")
    category_percentage(set(y_train), list(y_train))

    X_test, X_dev, y_test, y_dev = train_test_split(
        X_remaining, remaining_y, test_size=0.5
    )
    print("y_test")
    category_percentage(set(y_test), list(y_test))
    write_conllu_file(OUTPUT_DATA, "test", X_test)
    print("y_dev")
    category_percentage(set(y_dev), list(y_dev))
    write_conllu_file(OUTPUT_DATA, "dev", X_dev)


if __name__ == "__main__":
    load_data(OUTPUT_DATA)
