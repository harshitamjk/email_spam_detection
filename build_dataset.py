import os
import pandas as pd
from email import policy
from email.parser import BytesParser

def read_email(file_path):
    with open(file_path, "rb") as file:
        message = BytesParser(policy=policy.default).parse(file)

    subject = message["subject"]

    if message.is_multipart():
        body = ""

        for part in message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode("utf-8", errors="ignore")
                #body += part.get_content()
    else:
        payload = message.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="ignore")
        else:
            body = ""
        #body = message.get_content()
    return subject, body

def collect_emails(folder_path, label):
    data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                subject, body = read_email(file_path)

                data.append({
                    "subject": subject,
                    "body": body,
                    "label": label
                })

            except Exception as e:
                print(f"Could not read {filename}: {e}")

    return data

ham_emails = collect_emails(
    "dataset/easy-ham/easy_ham",
    0
)
print("Number of ham emails:", len(ham_emails))

ham_emails_2 = collect_emails(
    "dataset/easy-ham-2/easy_ham_2",
    0
)
print("Number of second ham emails:", len(ham_emails_2))

spam_emails = collect_emails(
    "dataset/spam/spam",
    1
)
print("Number of spam emails:", len(spam_emails))

spam_emails_2 = collect_emails(
    "dataset/spam-2/spam_2",
    1
)
print("Number of spam emails:", len(spam_emails_2))

all_emails = ham_emails + ham_emails_2 + spam_emails + spam_emails_2
print("Total emails:", len(all_emails))

df = pd.DataFrame(all_emails)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

df.to_csv("dataset/emails.csv", index=False)

print("\nDataset saved successfully!")