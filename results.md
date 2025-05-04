Summary of Results:
| Instance | Algorithm                    | Cost (min - avg - max) | Time (ms, avg) | Iterations (avg) |
|----------|------------------------------|------------------------|----------------|------------------|
| krob200 | MSLS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Iterations: 200) | 36278 (36771.20 - 37497) |        5120 - 37497) |        51920 - 37497) |        5196.30 |              N/A |
| krob200 | ILS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: SmallPerturbation(n_moves=10)) | 36529 (37046.50 - 37563) |        5210.20 |            199.4 |
| krob200 | LNS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 35703 (36925.10 - 37617) |        5209.60 |            181.5 |
| krob200 | LNSa (no LS after repair) (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 31426 (32409.20 - 33633) |        5196.60 |           1987.2 |
| kroa200 | MSLS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Iterations: 200) | 35945 (36745.20 - 37436) |        5379.90 |              N/A |
| kroa200 | ILS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: SmallPerturbation(n_moves=10)) | 36297 (36881.00 - 37396) |        5392.60 |            190.6 |
| kroa200 | LNS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 36325 (36943.60 - 37434) |        5396.10 |            178.1 |
| kroa200 | LNSa (no LS after repair) (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 31064 (32206.90 - 33293) |        5381.00 |           1796.7 |