import tarfile
import os
def extract_archive(archive_path, output_folder):
    with tarfile.open(archive_path, "r:bz2") as archive:
        archive.extractall(output_folder)

extract_archive(
    "dataset/20021010_spam.tar.bz2",
    "dataset/spam"
)

extract_archive(
    "dataset/20030228_easy_ham_2.tar.bz2",
    "dataset/easy-ham-2"
)

extract_archive(
    "dataset/20050311_spam_2.tar.bz2",
    "dataset/spam-2"
)