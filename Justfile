set dotenv-load

# Default resolution
res := "500"
smoothing := "20000.0"
construct := "uv run nzcvm basin main"
coastline := "resources/coastline.wkb.gz"
canterbury:
    @test -d models/Cant_Pliocene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 models/Cant_Pliocene.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v3_Pliocene_Enforced.fd_modfile -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}
    @test -d models/CantPaleogene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 models/CantPaleogene.zarr --rho 2.19 --vp 2.85 --vs 1.281 --priority 1 -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}
    @test -d models/BanksPeninsula.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5 models/BanksPeninsula.zarr --rho 5 --vp 5 --vs 5 -r {{ res }}
    @test -d models/CantMiocene.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 models/CantMiocene.zarr --rho 2.09 --vp 2.5 --vs 0.984 -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}

hanmer:
    @test -d models/Hanmer_Basin.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_outline_WGS84_v25p3.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_basement_WGS84_v25p3.h5 models/Hanmer_Basin.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 100

mackenzie:
    @test -d models/Mackenzie.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_basement_WGS84.h5 models/Mackenzie.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250

southland:
    @test -d models/Southland_Basin_1.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_1.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250
    @test -d models/Southland_Basin_2.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_2.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_2.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

west_coast:
    @test -d models/WestCoast.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_basement_WGS84.h5 models/WestCoast.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

te_anau:
    @test -d models/TeAnau.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_basement_WGS84.h5 models/TeAnau.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

balclutha:
    @test -d models/Balclutha.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_basement_WGS84.h5 models/Balclutha.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

castle_hill:
    @test -d models/CastleHill.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_basement_WGS84.h5 models/CastleHill.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

cheviot:
    @test -d models/Cheviot.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_basement_WGS84.h5 models/Cheviot.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

collingwood:
    @for i in 1 2 3; do \
        test -d models/Collingwood${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_basement_WGS84.h5 models/Collingwood${i}.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400; \
    done

dunedin:
    @test -d models/Dunedin.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_basement_WGS84.h5 models/Dunedin.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

alexandra:
    @test -d models/Alexandra.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_basement_WGS84.h5 models/Alexandra.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

gisborne:
    @test -d models/Gisborne.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_basement_WGS84.h5 models/Gisborne.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

hakataramea:
    @test -d models/Hakataramea.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_basement_WGS84.h5 models/Hakataramea.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

karamea:
    @test -d models/Karamea.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_basement_WGS84.h5 models/Karamea.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

marlborough:
    @test -d models/Marlborough.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_basement_WGS84.h5 models/Marlborough.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

mosgiel:
    @test -d models/Mosgiel.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_basement_WGS84.h5 models/Mosgiel.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

murchison:
    @test -d models/Murchison.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_basement_WGS84.h5 models/Murchison.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

ranfurly:
    @test -d models/Ranfurly.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_basement_WGS84.h5 models/Ranfurly.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

rarakau:
    @test -d models/Rarakau.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_basement_WGS84.h5 models/Rarakau.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

springs_junction:
    @test -d models/SpringsJunction.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_basement_WGS84.h5 models/SpringsJunction.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

tolaga_bay:
    @test -d models/TolagaBay.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_basement_WGS84.h5 models/TolagaBay.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waiapu:
    @test -d models/Waiapu.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_basement_WGS84.h5 models/Waiapu.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waikato_hauraki:
    @test -d models/WaikatoHauraki.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_basement_WGS84.h5 models/WaikatoHauraki.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

wairarapa:
    @test -d models/Wairarapa.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_basement_WGS84.h5 models/Wairarapa.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

waitaki:
    @test -d models/Waitaki.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_basement_WGS84.h5 models/Waitaki.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wakatipu:
    @test -d models/Wakatipu.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_basement_WGS84.h5 models/Wakatipu.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wanaka:
    @test -d models/Wanaka.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_basement_WGS84.h5 models/Wanaka.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

westport:
    @test -d models/Westport.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Westport/Westport_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Westport/Westport_basement_WGS84.h5 models/Westport.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

north_canterbury:
    @test -d models/NorthCanterbury.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_outline_WGS84_v19p1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_basement_WGS84_v19p1.h5 models/NorthCanterbury.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wellington:
    @test -d models/Wellington.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_basement_WGS84_v21p8.h5 models/Wellington.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

palmerston_north:
    @test -d models/PalmerstonNorth.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_basement_WGS84_v25p5.h5 models/PalmerstonNorth.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

omaio_bay:
    @for i in 1 2 3; do \
        test -d models/OmaioBay${i}.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_outline_WGS84_v22p3_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_basement_WGS84_v22p3.h5 models/OmaioBay${i}.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

nelson:
    @test -d models/Nelson.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_basement_WGS84_v25p5.h5 models/Nelson.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

kaikoura:
    @test -d models/Kaikoura.zarr || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_basement_WGS84.h5 models/Kaikoura.zarr --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

basins: canterbury hanmer mackenzie southland west_coast te_anau balclutha castle_hill cheviot collingwood dunedin alexandra gisborne hakataramea karamea marlborough mosgiel murchison ranfurly rarakau springs_junction tolaga_bay waiapu waikato_hauraki wairarapa waitaki wakatipu wanaka westport north_canterbury wellington palmerston_north omaio_bay nelson kaikoura

tomography := "uv run nzcvm tomography convert"
surface := "uv run nzcvm convert-tiff main"

ep2020:
    @test -f models/ep2020.zarr || {{ tomography }} ep2020.csv models/ep2020.zarr

vs30:
    @test -f resources/vs30.zarr || {{ surface }} ${VS30_TIFF} 1 resources/vs30.zarr --downsample 3 

pytest:
    uv run --dev --config-setting 'build-args=--profile=dev' pytest -s tests
    uv run --dev --config-setting 'build-args=--profile=dev' pytest --doctest-modules nzcvm/ -v

cargo:
    cargo test

test: pytest cargo

ty:
    uv run ty check nzcvm

ruff:
    uv run ruff format
    uv run ruff check --select I --fix

clippy:
    cargo clippy -- -D warnings

lint: ty ruff clippy
