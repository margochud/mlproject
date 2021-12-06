rule all:
    input:
        "../figures/accuracy_canser"
        "../figures/precision_canser"
        "../figures/recall_canser"
        "../figures/accuracy_spam"
        "../figures/precision_spam"
        "../figures/recall_spam"

rule cancer:
    input:
        "../data/canser.csv"
    output:
        "../figures/accuracy_canser"
        "../figures/precision_canser"
        "../figures/recall_canser"
    script:
        f"cli.py {input} {output}"

rule cancer:
    input:
        "../data/spam.csv"
    output:
        "../figures/accuracy_spam"
        "../figures/precision_spam"
        "../figures/recall_spam"
    script:
        f"cli.py {input} {output}"


