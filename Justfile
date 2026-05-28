set dotenv-load

# Default resolution
res := "500"
smoothing := "20000.0"
construct := "uv run nzcvm basin main"
coastline := "resources/coastline.wkb.gz"
canterbury:
    @test -f models/Cant_Pliocene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 models/Cant_Pliocene.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v3_Pliocene_Enforced.fd_modfile -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}
    @test -f models/CantPaleogene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 models/CantPaleogene.vtkhdf --rho 2.19 --vp 2.85 --vs 1.281 --priority 1 -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}
    @test -f models/BanksPeninsula.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5 models/BanksPeninsula.vtkhdf --rho 5 --vp 5 --vs 5 -r {{ res }}
    @test -f models/CantMiocene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 models/CantMiocene.vtkhdf --rho 2.09 --vp 2.5 --vs 0.984 -r {{ res }} --smoothing {{ smoothing }} --coastline {{ coastline }}

hanmer:
    @test -f models/Hanmer_Basin.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_outline_WGS84_v25p3.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_basement_WGS84_v25p3.h5 models/Hanmer_Basin.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 100

mackenzie:
    @test -f models/Mackenzie.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_basement_WGS84.h5 models/Mackenzie.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250

southland:
    @test -f models/Southland_Basin_1.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_1.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250
    @test -f models/Southland_Basin_2.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_2.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 models/Southland_Basin_2.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

west_coast:
    @test -f models/WestCoast.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_basement_WGS84.h5 models/WestCoast.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

te_anau:
    @test -f models/TeAnau.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_basement_WGS84.h5 models/TeAnau.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

balclutha:
    @test -f models/Balclutha.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_basement_WGS84.h5 models/Balclutha.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

castle_hill:
    @test -f models/CastleHill.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_basement_WGS84.h5 models/CastleHill.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

cheviot:
    @test -f models/Cheviot.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_basement_WGS84.h5 models/Cheviot.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

collingwood:
    @for i in 1 2 3; do \
        test -f models/Collingwood.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_basement_WGS84.h5 models/Collingwood${i}.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400; \
    done

dunedin:
    @test -f models/Dunedin.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_basement_WGS84.h5 models/Dunedin.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

alexandra:
    @test -f models/Alexandra.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_basement_WGS84.h5 models/Alexandra.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

gisborne:
    @test -f models/Gisborne.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_basement_WGS84.h5 models/Gisborne.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

hakataramea:
    @test -f models/Hakataramea.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_basement_WGS84.h5 models/Hakataramea.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

karamea:
    @test -f models/Karamea.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_basement_WGS84.h5 models/Karamea.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

marlborough:
    @test -f models/Marlborough.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_basement_WGS84.h5 models/Marlborough.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

mosgiel:
    @test -f models/Mosgiel.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_basement_WGS84.h5 models/Mosgiel.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

omaiobay:
    @test -f models/OmaioBay.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_basement_WGS84.h5 models/OmaioBay.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

murchison:
    @test -f models/Murchison.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_basement_WGS84.h5 models/Murchison.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

ranfurly:
    @test -f models/Ranfurly.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_basement_WGS84.h5 models/Ranfurly.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

rarakau:
    @test -f models/Rarakau.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_basement_WGS84.h5 models/Rarakau.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

springs_junction:
    @test -f models/SpringsJunction.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_basement_WGS84.h5 models/SpringsJunction.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

tolaga_bay:
    @test -f models/TolagaBay.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_basement_WGS84.h5 models/TolagaBay.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waiapu:
    @test -f models/Waiapu.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_basement_WGS84.h5 models/Waiapu.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waikato_hauraki:
    @test -f models/WaikatoHauraki.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_basement_WGS84.h5 models/WaikatoHauraki.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

wairarapa:
    @test -f models/Wairarapa.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_basement_WGS84.h5 models/Wairarapa.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

waitaki:
    @test -f models/Waitaki.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_basement_WGS84.h5 models/Waitaki.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wakatipu:
    @test -f models/Wakatipu.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_basement_WGS84.h5 models/Wakatipu.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wanaka:
    @test -f models/Wanaka.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_basement_WGS84.h5 models/Wanaka.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

westport:
    @test -f models/Westport.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Westport/Westport_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Westport/Westport_basement_WGS84.h5 models/Westport.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

north_canterbury:
    @test -f basins/NorthCanterbury.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_outline_WGS84_v19p1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/NorthCanterbury/NorthCanterbury_basement_WGS84_v19p1.h5 basins/NorthCanterbury.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wellington:
    @test -f basins/Wellington.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wellington/Wellington_basement_WGS84_v21p8.h5 basins/Wellington.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

palmerston_north:
    @test -f basins/PalmerstonNorth.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/PalmerstonNorth/PalmerstonNorth_basement_WGS84_v25p5.h5 basins/PalmerstonNorth.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

omaio_bay:
    @for i in 1 2 3; do \
        test -f basins/OmaioBay${i}.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_outline_WGS84_v22p3_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/OmaioBay/OmaioBay_basement_WGS84_v22p3.h5 basins/OmaioBay${i}.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}; \
    done

nelson:
    @test -f basins/Nelson.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Nelson/Nelson_basement_WGS84_v25p5.h5 basins/Nelson.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

kaikoura:
    @test -f basins/Kaikoura.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Kaikoura/Kaikoura_basement_WGS84.h5 basins/Kaikoura.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400 --smoothing {{ smoothing }} --coastline {{ coastline }}

basins: canterbury hanmer mackenzie southland west_coast te_anau balclutha castle_hill cheviot collingwood dunedin alexandra gisborne hakataramea karamea marlborough mosgiel murchison ranfurly rarakau springs_junction tolaga_bay waiapu waikato_hauraki wairarapa waitaki wakatipu wanaka westport north_canterbury wellington palmerston_north omaio_bay nelson kaikoura

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
