import sys

batch = sys.argv[1]

# Specify the command to run in Swiss Army Knife
cmd=f"tar -chzvf waveforms_{batch}.tar.gz *.npy"

# Generate a json file with the list of input files, bash command, docker image, and other parameters
json_snippets = []
for dx_file in open(f"x{str(batch).zfill(2)}", "r").readlines():
    project=dx_file.split(":")[0].strip()
    file_obj=dx_file.split(":")[1].strip()
    json_snippets.append('{"$dnanexus_link": {"project": "' + project + '","id": "' + file_obj + '"}}')

json_snippets = ",".join(json_snippets)
input_json = '{"cmd": "' + cmd + '","in": ['+ json_snippets +'],"image":"cicirello/alpine-plus-plus","mount_inputs":true}'

# Write json to file
with open(f"zip_ecgs.json", "w") as outfile:
    outfile.write(input_json)
