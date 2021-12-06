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
        f"CLI_canser.py {input} {output}"

rule spam:
    input:
        "../data/spam.csv"
    output:
        "../figures/accuracy_spam"
        "../figures/precision_spam"
        "../figures/recall_spam"
    script:
        f"CLI_spam.py {input} {output}"


