# Legacy Dataset Leakage Audit

Generated deterministically by `scripts/audit_legacy_dataset.py`.
This report uses exact SHA-256 content hashes; it makes no near-duplicate claims.

## Verdict

- Current evaluation trustworthy: **false**
- Dataset suitable for retraining: **false**
- Audit exit code: `1`

## Summary

| Measure | Count |
| --- | ---: |
| Inventoried files | 128 |
| Image files | 127 |
| Source image files | 74 |
| Readable image files | 127 |
| Unreadable image files | 0 |
| Unsupported files | 1 |
| Generated/runtime files | 53 |
| Extension/content mismatches | 5 |
| Exact duplicate groups | 34 |
| Cross-partition duplicate groups | 32 |
| Training/test overlap images | 10 |

## Training class counts

| Class | Images |
| --- | ---: |
| `dispense_error` | 9 |
| `misalignment` | 1 |
| `missing_part` | 4 |
| `no_defect` | 18 |

## Current sources and partitions

| Source | Image files |
| --- | ---: |
| `annotated_outputs` | 21 |
| `legacy_uploads` | 32 |
| `raw_source` | 32 |
| `test` | 10 |
| `training` | 32 |

Validation partition found: **no**.

## Blocking issues

- `cross_partition_duplicates` (32): Exact source-image content occurs in more than one partition/source.
- `training_test_overlap` (10): Test images are exact byte-for-byte matches of training images.
- `unsupported_source_files` (1): Unsupported files are present inside source-data locations.
- `significant_class_imbalance` (4): Training class imbalance exceeds the documented 3:1 audit threshold.

## Exact training/test overlap

- SHA-256 `072701ea602fab97871a637a1e578a1a43adb282a4a1d7db910a68db6f31b0cf`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_712.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_712.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_712.jpg`
- SHA-256 `2c1784ace4d22250d07d6baa35a7b2a1976d3604f8ee98b8b34cbae2b90f632b`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_715.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_715.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_715.jpg`
- SHA-256 `4327397751c861a38077c880654c0703ade79a53124608e5600543ba1e2bd8bb`
  - `backend/dataset/dispense_error/110219_143457.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143457.bmp`
  - `backend/test/dispense_error_110219_143457.bmp`
- SHA-256 `4fc125b25c047b3b85019c9f3fc5cbda06fb6f937d962b5f3521e59451f149bf`
  - `backend/dataset/no_defect/110219_142056.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142056.bmp`
  - `backend/test/no_defect_110219_142056.bmp`
- SHA-256 `5caae12ba7cd46d0e0b92578e4861278ee7be7805d1db998442256d3d1e75faf`
  - `backend/dataset/misalignment/1970_01_01NGripper_2_FAIL_722.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/NG/1970_01_01NGripper_2_FAIL_722.jpg`
  - `backend/test/misalignment_1970_01_01NGripper_2_FAIL_722.jpg`
- SHA-256 `64b079db064be20ac0c8551165ab21d22fa8bff9060189ca7128904ac66d632c`
  - `backend/dataset/dispense_error/110219_143222.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143222.bmp`
  - `backend/test/dispense_error_110219_143222.bmp`
- SHA-256 `6a0c8c45a1553d939ab8486f5dae5d61b10353a831249e98cff27e603a453a29`
  - `backend/dataset/missing_part/2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
  - `backend/test/missing_part_2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
- SHA-256 `8e20539e6fa7f3f07dda67f38ede67f6db36290e8d47929d48f12169bfd51ad2`
  - `backend/dataset/dispense_error/110219_143006.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143006.bmp`
  - `backend/test/dispense_error_110219_143006.bmp`
- SHA-256 `a624c1d18d837e1e16963a68e9d986cb437d41ef81521dd4276c2a2d041d7a24`
  - `backend/dataset/missing_part/POS4.69GRADOS.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/POS4.69GRADOS.bmp`
  - `backend/test/missing_part_POS4.69GRADOS.bmp`
- SHA-256 `cba05806cbcd0c3e4b8b86a9dfef090dc5f35a6425340e7b824070e72476465c`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_711.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_711.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_711.jpg`

## Cross-partition duplicate groups

- SHA-256 `072701ea602fab97871a637a1e578a1a43adb282a4a1d7db910a68db6f31b0cf`; partitions: `raw_source, test, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_712.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_712.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_712.jpg`
- SHA-256 `15afe29e08ea469a4fb2d3b55bc761b117d2dbcbcbcdf887d0e1330b4e082b46`; partitions: `raw_source, training`
  - `backend/dataset/missing_part/POS2_240GRADOS.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/POS2_240GRADOS.bmp`
- SHA-256 `1d22831a4d90512eb10f2f9117125065195a66f1d345f3ea8a70015cba6a8909`; partitions: `raw_source, training`
  - `backend/dataset/missing_part/2025_04_24_10_48_59N_FAIL_Climp_78832.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/2025_04_24_10_48_59N_FAIL_Climp_78832.bmp`
- SHA-256 `2c1784ace4d22250d07d6baa35a7b2a1976d3604f8ee98b8b34cbae2b90f632b`; partitions: `raw_source, test, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_715.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_715.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_715.jpg`
- SHA-256 `4327397751c861a38077c880654c0703ade79a53124608e5600543ba1e2bd8bb`; partitions: `raw_source, test, training`
  - `backend/dataset/dispense_error/110219_143457.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143457.bmp`
  - `backend/test/dispense_error_110219_143457.bmp`
- SHA-256 `4fc125b25c047b3b85019c9f3fc5cbda06fb6f937d962b5f3521e59451f149bf`; partitions: `raw_source, test, training`
  - `backend/dataset/no_defect/110219_142056.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142056.bmp`
  - `backend/test/no_defect_110219_142056.bmp`
- SHA-256 `562be0a98dbbf54366b55889adcd3e0e719f31c43a411303f6342e81e4ee3204`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/2025_04_24_10_53_23N_FAIL_Climp_1.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/2025_04_24_10_53_23N_FAIL_Climp_1.jpg`
- SHA-256 `58c6329462dd16eae2c7e1cedb97598707eb9089e5a6e78a21f4bec738216d50`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/110219_142623.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142623.bmp`
- SHA-256 `58e57cc767bb1e01093c3d51ab09f35381b3996327776471b389a52dc6b0b944`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_143047.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143047.bmp`
- SHA-256 `5a0af93347899b81e0127aa7e0ceeff480bb482307be2a0f96b4e3c4c4d1a919`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_980.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_980.jpg`
- SHA-256 `5caae12ba7cd46d0e0b92578e4861278ee7be7805d1db998442256d3d1e75faf`; partitions: `raw_source, test, training`
  - `backend/dataset/misalignment/1970_01_01NGripper_2_FAIL_722.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/NG/1970_01_01NGripper_2_FAIL_722.jpg`
  - `backend/test/misalignment_1970_01_01NGripper_2_FAIL_722.jpg`
- SHA-256 `64b079db064be20ac0c8551165ab21d22fa8bff9060189ca7128904ac66d632c`; partitions: `raw_source, test, training`
  - `backend/dataset/dispense_error/110219_143222.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143222.bmp`
  - `backend/test/dispense_error_110219_143222.bmp`
- SHA-256 `6a0c8c45a1553d939ab8486f5dae5d61b10353a831249e98cff27e603a453a29`; partitions: `raw_source, test, training`
  - `backend/dataset/missing_part/2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
  - `backend/test/missing_part_2025_04_24_12_12_52_N_CLIMP_1_FAIL_849.jpg`
- SHA-256 `6fa1be4d9a232a35f8f0fa61cee4d6982d16981d75122108c556a3ed6ad97cd7`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/2025_04_24_12_13_04_N_CLIMP_1_OK_850.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/2025_04_24_12_13_04_N_CLIMP_1_OK_850.jpg`
- SHA-256 `7a246e2e663812741c564ce5c64e1dec452180ba036d5d6bd12895e3afccb043`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_143144.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143144.bmp`
- SHA-256 `7cfeb0cb84c4c7e7515c86f845ded340f5671dc287b0233ef7998c7f628e3c08`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/1970_01_01_03_14_01N_FAIL_Climp_1827.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/1970_01_01_03_14_01N_FAIL_Climp_1827.jpg`
- SHA-256 `8e20539e6fa7f3f07dda67f38ede67f6db36290e8d47929d48f12169bfd51ad2`; partitions: `raw_source, test, training`
  - `backend/dataset/dispense_error/110219_143006.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143006.bmp`
  - `backend/test/dispense_error_110219_143006.bmp`
- SHA-256 `9314443449cf156ec434c4ffb3c16cdc68e53b834007b2f30c041610f464151a`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/110219_142134.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142134.bmp`
- SHA-256 `9f5d9a7a613daec9237ba5b5ae8c54fe679b7fb34522b094cdb5ae58c83ece1c`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/2025_04_24_12_13_04_N_CLIMP_1_OK_851.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/2025_04_24_12_13_04_N_CLIMP_1_OK_851.bmp`
- SHA-256 `a3d3d2c8cbce563fa92074d271d0ecbd6090fa5840b68c08092018ae91b3f7ab`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_142725.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_142725.bmp`
- SHA-256 `a624c1d18d837e1e16963a68e9d986cb437d41ef81521dd4276c2a2d041d7a24`; partitions: `raw_source, test, training`
  - `backend/dataset/missing_part/POS4.69GRADOS.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/NOK/POS4.69GRADOS.bmp`
  - `backend/test/missing_part_POS4.69GRADOS.bmp`
- SHA-256 `a8f7279a39466ddc6854c74c6098cd9b14782ef4caccc787e256c317799445a1`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_142915.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_142915.bmp`
- SHA-256 `bb5d6bafcd01968f5a5aa95ae3b2bbae6465285b631a57bd0e9d73e7181b762a`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/2025_04_24_12_13_04_N_CLIMP_1_OK_852.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/2025_04_24_12_13_04_N_CLIMP_1_OK_852.bmp`
- SHA-256 `c26659efcfed50329be2dbfa8946c85665d226544edfa0326e1944056cdea7eb`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_718.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_718.jpg`
- SHA-256 `c38929e811d0a8cb460330e12a18283bb800f83aede9beb0b605f87c912bcb98`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/110219_142059.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142059.bmp`
- SHA-256 `c6e6b3f880baf577ecd5ae60a841bb426ebe4e32c89a83e0a92cb0d91caef38b`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_143341.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143341.bmp`
- SHA-256 `cba05806cbcd0c3e4b8b86a9dfef090dc5f35a6425340e7b824070e72476465c`; partitions: `raw_source, test, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_711.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_711.jpg`
  - `backend/test/no_defect_1970_01_01NGripper_2_FAIL_711.jpg`
- SHA-256 `e1fa1d80a167e694ce69a6132b9bdfca654190999dc322193f14a858413e6586`; partitions: `raw_source, training`
  - `backend/dataset/dispense_error/110219_143613.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/NG/110219_143613.bmp`
- SHA-256 `e75916e25e048cabb44db28e452ea9b0c580bc4d3e4ac709dc073814f39e08a7`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/2025_04_24_10_55_04N_FAIL_Climp_4.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Crimp/OK/2025_04_24_10_55_04N_FAIL_Climp_4.bmp`
- SHA-256 `ef583b0556cb594c26e39db24939388191862b88079557110cdd8a195df2904b`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_976.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_976.jpg`
- SHA-256 `ef6d2eecc3db3eaedad0265f47830dd0e1978446c5a8d43894505b15cfa7e562`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/1970_01_01NGripper_2_FAIL_714.jpg`
  - `backend/dataset_raw/Imagenes OK& NOK/PINS/OK/1970_01_01NGripper_2_FAIL_714.jpg`
- SHA-256 `efd1fc1905e397e5ccb8268432edb8b4d7e602f7b3d6ffbf9645274b6ebc2590`; partitions: `raw_source, training`
  - `backend/dataset/no_defect/110219_142105.bmp`
  - `backend/dataset_raw/Imagenes OK& NOK/Dispensing/OK/110219_142105.bmp`

## Unexpected, unreadable, and unsupported files

### Unexpected class directories
- None

### Unreadable image files
- None

### Unsupported files
- `backend/dataset/dataset.zip`

### Extension/content mismatches
- `backend/uploads/temp_1fdb0b2f4a2a4bd48dcf609e26d3b2dd.jpg`
- `backend/uploads/temp_27bbfd6402964b4eaa7dcae1783f7718.jpg`
- `backend/uploads/temp_586d36ff90c94d79b3b029ecd3d92a9a.jpg`
- `backend/uploads/temp_d690ba7607d14bf2b3a14e8509419cc8.jpg`
- `backend/uploads/temp_d918056d52054642a19b918adaff662d.jpg`

### Empty class directories
- None

### Images outside the canonical class contract
- None

## Duplicate filenames with different content

- None

## Grouping evidence

- Proven production groups: 0
- Source images requiring human grouping review: 74
- Filename date/timestamp-like prefixes are heuristic hints only; no board, lot, panel, session, or batch grouping is proven.

## Conclusion

The current evaluation is not trustworthy because blocking leakage or dataset-integrity issues exist.
The dataset is not ready for retraining. Resolve every recorded blocker and obtain reviewed production grouping metadata before splitting.
