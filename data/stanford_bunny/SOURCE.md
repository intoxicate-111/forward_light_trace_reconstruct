# Stanford Bunny data notice

The v0.3d experiment uses `bun_zipper.ply` from the Stanford 3D Scanning
Repository:

- Source: Stanford University Computer Graphics Laboratory
- Dataset page: https://graphics.stanford.edu/data/3Dscanrep/
- Official archive: https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
- Archive SHA-256: `a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84`
- Mesh SHA-256: `b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe`
- Original mesh: 35,947 vertices and 69,451 triangles, with five bottom holes

Stanford permits research use and free redistribution of these models with
acknowledgement, but restricts commercial use without permission. The mesh is
therefore downloaded and hash-verified at run time rather than distributed
under this repository's MIT license. Install the optional dependencies with
`python -m pip install -e '.[bunny]'`, then use `python demo.py --bunny`.

The experiment evaluates geometry against the original mesh. A deterministic
five-cap watertight proxy is created only for fixed sign determination; it is
not committed and does not replace the evaluation reference.
