def cpop_pinyin2ph_func():
    # In the README file of opencpop dataset, they defined a "pinyin to phoneme mapping table"
    pinyin2phs = {'AP': 'AP', 'SP': 'SP'}
    with open('NeuralSeq/inference/svs/opencpop/cpop_pinyin2ph.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split('|') if x.strip() != '']
            pinyin2phs[elements[0]] = elements[1]
    return pinyin2phs