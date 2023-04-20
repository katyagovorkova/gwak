rule generate_background:
    input:
        script = 'generate.py',
    params:
        stype = 'background'
    output:
        file = 'output/background_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_bbh:
    input:
        script = 'generate.py',
    params:
        stype = 'bbh'
    output:
        file = 'output/bbh_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_sg:
    input:
        script = 'generate.py',
    params:
        stype = 'sg'
    output:
        file = 'output/sg_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_glitch:
    input:
        script = 'generate.py',
    params:
        stype = 'glitch'
    output:
        file = 'output/glitch_segs.npy'
    shell:
        'python {input.script} {output.file} {params.stype}'

rule generate_data:
    output:
        rules.generate_background.output,
        rules.generate_bbh.output,
        rules.generate_sg.output,
        rules.generate_glitch.output

# rule train_quak:
#     input:
#     output:
#     shell:

# rule calculate_pearson:
#     input:
#     output:
#     shell:

# rule calculate_metric:
#     input:
#     output:
#     shell:

# rule plot_results:
#     input:
#     output:
#     shell: