import csv


def parse_vcf_2_csv(vcf_file_path: str, output_fila_path: str):
    with open(vcf_file_path, 'r') as file:
        data = []
        for line in file:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    header = line.strip().split('\t')
            else:
                fields = line.strip().split('\t')
                chrom, pos, vid, ref, alt, qual, fltr, info, fmt = fields[:9]
                samples = fields[9:]
                fmt_keys = fmt.split(':')

                # Process each sample
                for sample, sample_data in zip(header[9:], samples):
                    sample_values = sample_data.split(':')
                    sample_info = dict(zip(fmt_keys, sample_values))
                    data.append({
                        'Sample': sample,
                        'Chromosome': chrom,
                        'Position': pos,
                        'Reference': ref,
                        'Alternate': alt,
                        'INFO': sample_info.keys(),
                        'Genotype': sample_info.get('GT'),
                        'List of Phred-scaled genotype likelihoods': sample_info.get('PL'),
                        'Raw read depth': sample_info.get('DP'),
                    })

    with open(output_fila_path, 'w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f'Parsed VCF data saved to {output_fila_path}')
