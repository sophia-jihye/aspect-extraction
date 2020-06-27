import os, csv
import pandas as pd 
from ReviewSentence import ReviewSentence
import stanfordnlp

nlp = stanfordnlp.Pipeline()

def parse_sentence(sentence, targets):
    targets = [item.split(' ') for item in targets]
    targets = [val for sublist in targets for val in sublist]
    parsed_sent = sentence.dependencies
    vals_list = []
    for i in range(len(parsed_sent)):
        token = parsed_sent[i][2].text
        target_type_ = 'O'
        if token in targets: target_type_ = 'B-A'
        if i>0 and target_type_ == 'B-A' and parsed_sent[i-1][2].text in targets: target_type_ = 'I-A'
        vals_list.append([token, parsed_sent[i][2].xpos, target_type_])
    return vals_list

def parse_df(train_or_test, df, filename):
    filepath = os.path.join('parsed', '%s_%s_%d.iob' % (filename, train_or_test, len(df)))
    text_file = open(filepath, "w", encoding='utf-8')
    for _, df_row in df.iterrows():
        sample = df_row['content']
        sample_target = df_row['target']
        doc = nlp(sample)
        for sentence in doc.sentences:
            vals_list = parse_sentence(sentence, sample_target)
            for vals in vals_list:
                if vals is None: 
                    continue
                text_file.write('\t'.join(vals)+'\n')
        text_file.write('\n')
    text_file.close()
    print('Created %s' % filepath)
    
def main():
    dfs = []
    versions = ['three']
    for version in versions:
        source_dir = os.path.join('raw', version)
        for filename in os.listdir(source_dir):
            if filename == "Readme.txt":
                continue

            rows = []
            filepath = os.path.join(source_dir, filename)
            print("Processing %s.." % filepath)

            with open(filepath, encoding="utf-8") as lines:
                for ln in lines:
                    line = ln.strip().replace("\t", " ")
                    r = ReviewSentence.parse(line)
                    if r is not None:
                        if r.sentence_type == 'review':
                            rows.append(r.to_row())
            df = pd.DataFrame.from_records(rows, columns=("content", "target"))
            df['filename'] = '-'.join([version, filename.replace('.txt', '')])
            dfs.append(df)

    concat_df = pd.concat(dfs, ignore_index=True)

    before_ = len(concat_df)
    concat_df = concat_df.drop_duplicates(subset=['content'])
    print('# of duplicates:', before_ - len(concat_df))
    
    for filename in concat_df['filename'].unique():
        print('[%s] Parsing to B-A|I-A format..' % filename)
        df = concat_df[concat_df['filename']==filename]
        test_len = round(len(df)/10)
        test_df = df.iloc[:test_len]
        parse_df('Test', test_df, filename)

        train_df = df.iloc[test_len:]
        parse_df('Train', train_df, filename)
        
if __name__ == '__main__':
    main()