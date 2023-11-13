
from classify import train_model, finetune_xlm_roberta
from tqdm import tqdm
from numpy import mean, std

with open("res2.txt", "w") as f:
    
    # get xlm-r results
    acc, f1_s, f1_l, ood_acc = [], [], [], []
    acc2, f1_s2, f1_l2 = [], [], []
    
    for _ in range(5):
        cr = finetune_xlm_roberta(
            lr=2e-5,
            epochs=4,
            train_on="regdata",
            test_on="both",
            augment=True
        )
        acc.append(cr['accuracy'])
        f1_s.append(cr['colloquial']['f1-score'])
        f1_l.append(cr['literary']['f1-score'])
        print(cr)

        cr = finetune_xlm_roberta(
            lr=2e-5,
            epochs=4,
            train_on="regdata",
            test_on="dakshina",
            augment=True
        )
        ood_acc.append(cr['accuracy'])
        print(cr)
            
        cr = finetune_xlm_roberta(
            lr=2e-5,
            epochs=4,
            train_on="both",
            test_on="both",
            augment=True
        )
        acc2.append(cr['accuracy'])
        f1_s2.append(cr['colloquial']['f1-score'])
        f1_l2.append(cr['literary']['f1-score'])
        print(cr)
    
    fields = [
        "xlm-r",
        f"${mean(acc):.1%}$",
        f"${mean(f1_s):.3f}$",
        f"${mean(f1_l):.3f}$",
        f"${mean(ood_acc):.1%}$",
        f"${mean(acc2):.1%}$",
        f"${mean(f1_s2):.3f}$",
        f"${mean(f1_l2):.3f}$",
    ]

    string = " & ".join(fields).replace("%", "\\%")
    f.write(string)
    print(string)

    for model in ["gnb", "mnb"]:
        for word in range(1, -1, -1):
            for char in range(4, -1, -1):
                if char == 0 and word == 0:
                    continue
                
                acc, f1_s, f1_l, ood_acc = [], [], [], []
                acc2, f1_s2, f1_l2 = [], [], []

                for _ in tqdm(range(5)):
                    gnb, y_test, y_pred, cr, X_train, y_train, X_test, y_test, label_to_id, id_to_label = train_model(
                        model=model,
                        char_n_max=char,
                        word_n_max=word,
                        train_on="regdata",
                        test_on="both",
                        augment=True
                    )
                    acc.append(cr['accuracy'])
                    f1_s.append(cr['colloquial']['f1-score'])
                    f1_l.append(cr['literary']['f1-score'])

                X_train, y_train, X_test, y_test, label_to_id, id_to_label = None, None, None, None, None, None

                for _ in tqdm(range(5)):
                    gnb, y_test, y_pred, cr2, X_train, y_train, X_test, y_test, label_to_id, id_to_label = train_model(
                        model=model,
                        char_n_max=char,
                        word_n_max=word,
                        train_on="regdata",
                        test_on="dakshina",
                        augment=True
                    )
                    ood_acc.append(cr2['accuracy'])

                for _ in tqdm(range(5)):
                    gnb, y_test, y_pred, cr2, X_train, y_train, X_test, y_test, label_to_id, id_to_label = train_model(
                        model=model,
                        char_n_max=char,
                        word_n_max=word,
                        train_on="both",
                        test_on="both",
                        augment=True
                    )
                    acc2.append(cr['accuracy'])
                    f1_s2.append(cr['colloquial']['f1-score'])
                    f1_l2.append(cr['literary']['f1-score'])

    
                fields = [
                    model,
                    f"$c={char}, w={word}$",
                    f"${mean(acc):.1%} \pm {std(acc):.1%}$",
                    f"${mean(f1_s):.3f} \pm {std(f1_s):.3f}$",
                    f"${mean(f1_l):.3f} \pm {std(f1_l):.3f}$",
                    f"${mean(ood_acc):.1%} \pm {std(ood_acc):.1%}$",
                    f"${mean(acc2):.1%} \pm {std(acc2):.1%}$",
                    f"${mean(f1_s2):.3f} \pm {std(f1_s2):.3f}$",
                    f"${mean(f1_l2):.3f} \pm {std(f1_l2):.3f}$",
                ]

                string = " & ".join(fields).replace("%", "\\%")
                f.write(string)
                print(string)