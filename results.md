Summary of Results:
| Instance | Algorithm                    | Cost (min - avg - max) | Time (ms, avg) | Iterations (avg) |
|----------|------------------------------|------------------------|----------------|------------------|
| krob200 | MSLS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Iterations: 200) | 36237 (36957.30 - 37474) |        5679.10 |              N/A |
| krob200 | ILS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: SmallPerturbation(n_moves=10)) | 36351 (36947.00 - 37580) |        5692.30 |            192.7 |
| krob200 | LNS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 36506 (37082.00 - 37678) |        5695.70 |            176.2 |
| krob200 | LNSa (no LS after repair) (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 31598 (32315.50 - 32825) |        5680.10 |           1902.4 |
| kroa200 | MSLS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Iterations: 200) | 35608 (36599.90 - 37296) |        9332.70 |              N/A |
| kroa200 | ILS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: SmallPerturbation(n_moves=10)) | 35795 (36500.70 - 37219) |        9365.10 |            174.8 |
| kroa200 | LNS (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 36233 (36771.90 - 37456) |        9358.00 |            165.3 |
| kroa200 | LNSa (no LS after repair) (Base: Local Search (Candidate k=10, EdgeExchange, Init: Random), Perturb: LargePerturbation(destroy=0.20)) (LS on Initial) | 30832 (32111.00 - 33452) |        9334.90 |           1922.8 |