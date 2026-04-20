set dotenv-load := true

canterbury:
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 basins/Cant_Pliocene.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v3_Pliocene_Enforced.fd_modfile -r 500

    # Canterbury Paleogene
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 basins/CantPaleogene.vtkhdf --rho 2.19 --vp 2.85 --vs 1.281 --priority 1 -r 500

    # Banks Peninsula Volcanics
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5 basins/BanksPeninsula.vtkhdf --rho 5 --vp 5 --vs 5 -r 500

    # Canterbury Miocene
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 basins/CantMiocene.vtkhdf --rho 2.09 --vp 2.5 --vs 0.984 -r 500

hanmer:
    # Hanmer Basin
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_outline_WGS84_v25p3.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_basement_WGS84_v25p3.h5 basins/Hanmer_Basin.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 100

mackenzie:
    # Mackenzie
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_basement_WGS84.h5 basins/Mackenzie.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250

southland:
    # Southland Basin 1
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 basins/Southland_Basin_1.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250

    # Southland Basin 2
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_2.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 basins/Southland_Basin_2.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

west_coast:
    # West Coast
    uv run scripts/construct_mesh.py ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_basement_WGS84.h5 basins/WestCoast.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

basins: canterbury hanmer mackenzie southland west_coast
