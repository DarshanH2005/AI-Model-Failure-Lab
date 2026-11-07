# Waterbirds external validation results

Separate photographic-composite bird classification using official splits. This does not validate industrial defect detection or intervention-repair transfer. No foreground edits were applied.

| Architecture | Seed | Unweighted accuracy | Adjusted accuracy | Worst-group accuracy | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|
| resnet18.a1_in1k | 11 | 0.8138 | 0.9596 | 0.3349 | 0.0891 | 0.0251 |
| resnet18.a1_in1k | 22 | 0.8067 | 0.9605 | 0.3738 | 0.0914 | 0.0254 |
| resnet18.a1_in1k | 33 | 0.7939 | 0.9581 | 0.3676 | 0.1082 | 0.0341 |
| deit_tiny_patch16_224.fb_in1k | 11 | 0.8112 | 0.9565 | 0.5561 | 0.1267 | 0.0404 |
| deit_tiny_patch16_224.fb_in1k | 22 | 0.8196 | 0.9585 | 0.3925 | 0.1231 | 0.0416 |
| deit_tiny_patch16_224.fb_in1k | 33 | 0.8278 | 0.9637 | 0.5234 | 0.1330 | 0.0482 |

## Training-seed variability

- resnet18.a1_in1k, accuracy: mean 0.8048, sample SD 0.0101 (3 seeds).
- resnet18.a1_in1k, adjusted_accuracy: mean 0.9594, sample SD 0.0012 (3 seeds).
- resnet18.a1_in1k, worst_group_accuracy: mean 0.3588, sample SD 0.0209 (3 seeds).
- deit_tiny_patch16_224.fb_in1k, accuracy: mean 0.8195, sample SD 0.0083 (3 seeds).
- deit_tiny_patch16_224.fb_in1k, adjusted_accuracy: mean 0.9596, sample SD 0.0038 (3 seeds).
- deit_tiny_patch16_224.fb_in1k, worst_group_accuracy: mean 0.4907, sample SD 0.0865 (3 seeds).

## Group counts and uncertainty

- resnet18.a1_in1k-seed11: groups {'00': {'count': 2255, 'accuracy': 0.9933481152993349}, '01': {'count': 2255, 'accuracy': 0.7405764966740577}, '10': {'count': 642, 'accuracy': 0.3348909657320872}, '11': {'count': 642, 'accuracy': 0.9190031152647975}}; worst-group object-bootstrap 95% interval [0.29729391327888827, 0.37130935549922].
- resnet18.a1_in1k-seed22: groups {'00': {'count': 2255, 'accuracy': 0.9937915742793791}, '01': {'count': 2255, 'accuracy': 0.7090909090909091}, '10': {'count': 642, 'accuracy': 0.37383177570093457}, '11': {'count': 642, 'accuracy': 0.9252336448598131}}; worst-group object-bootstrap 95% interval [0.33642456087104977, 0.40930251222516706].
- resnet18.a1_in1k-seed33: groups {'00': {'count': 2255, 'accuracy': 0.991130820399113}, '01': {'count': 2255, 'accuracy': 0.6798226164079822}, '10': {'count': 642, 'accuracy': 0.367601246105919}, '11': {'count': 642, 'accuracy': 0.9283489096573209}}; worst-group object-bootstrap 95% interval [0.329803328290469, 0.4051610679679054].
- deit_tiny_patch16_224.fb_in1k-seed11: groups {'00': {'count': 2255, 'accuracy': 0.9880266075388027}, '01': {'count': 2255, 'accuracy': 0.6753880266075388}, '10': {'count': 642, 'accuracy': 0.5560747663551402}, '11': {'count': 642, 'accuracy': 0.9221183800623053}}; worst-group object-bootstrap 95% interval [0.5181626165474088, 0.5934729804756177].
- deit_tiny_patch16_224.fb_in1k-seed22: groups {'00': {'count': 2255, 'accuracy': 0.9942350332594235}, '01': {'count': 2255, 'accuracy': 0.7414634146341463}, '10': {'count': 642, 'accuracy': 0.3925233644859813}, '11': {'count': 642, 'accuracy': 0.9080996884735203}}; worst-group object-bootstrap 95% interval [0.3532809375828094, 0.42947354161074847].
- deit_tiny_patch16_224.fb_in1k-seed33: groups {'00': {'count': 2255, 'accuracy': 0.9924611973392461}, '01': {'count': 2255, 'accuracy': 0.719290465631929}, '10': {'count': 642, 'accuracy': 0.5233644859813084}, '11': {'count': 642, 'accuracy': 0.9345794392523364}}; worst-group object-bootstrap 95% interval [0.4811775937871057, 0.5599384858044164].

Adjusted accuracy weights test-group accuracies by official training-group proportions. Unweighted accuracy uses the actual test distribution. Bootstrap intervals describe test-object sampling, not all training variability. All six seeds/architectures were fixed before test evaluation. The finite 1500-step training budget and lack of valid masks limit conclusions; these are not published benchmark replications. See frozen protocol for preprocessing, known label issues and license sources.
## Interpretation

- deit_tiny_patch16_224.fb_in1k: mean unweighted test accuracy 81.95%, mean worst-group accuracy 49.07%; gap 32.89%.
- resnet18.a1_in1k: mean unweighted test accuracy 80.48%, mean worst-group accuracy 35.88%; gap 44.60%.
The lower group-specific performance demonstrates that high average performance can conceal failures under a changed background distribution. The scores alone do not prove an internal causal mechanism. This study did not test the targeted repair method on photographic images because valid foreground interventions were unavailable. The next repair-transfer study requires validated bird masks and a new protected evaluation protocol; no new training is selected from these test outcomes.