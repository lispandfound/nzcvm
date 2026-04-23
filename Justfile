set dotenv-load

# Default resolution
res := "500"

construct := "uv run scripts/construct_mesh.py"

canterbury:
    @test -f basins/Cant_Pliocene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 basins/Cant_Pliocene.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v3_Pliocene_Enforced.fd_modfile -r {{ res }}
    @test -f basins/CantPaleogene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Paleogene_WGS84.h5 basins/CantPaleogene.vtkhdf --rho 2.19 --vp 2.85 --vs 1.281 --priority 1 -r {{ res }}
    @test -f basins/BanksPeninsula.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_Miocene_WGS84.h5 ${NZCVM_DATA_ROOT}/regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5 basins/BanksPeninsula.vtkhdf --rho 5 --vp 5 --vs 5 -r {{ res }}
    @test -f basins/CantMiocene.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/regional/Canterbury/CantDEM.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Pliocene_46_WGS84_v8p9p18.h5 ${NZCVM_DATA_ROOT}/regional/Canterbury/Canterbury_Miocene_WGS84.h5 basins/CantMiocene.vtkhdf --rho 2.09 --vp 2.5 --vs 0.984 -r {{ res }}

hanmer:
    @test -f basins/Hanmer_Basin.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_outline_WGS84_v25p3.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hanmer/Hanmer_basement_WGS84_v25p3.h5 basins/Hanmer_Basin.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 100

mackenzie:
    @test -f basins/Mackenzie.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mackenzie/Mackenzie_basement_WGS84.h5 basins/Mackenzie.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250

southland:
    @test -f basins/Southland_Basin_1.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_1.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 basins/Southland_Basin_1.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 250
    @test -f basins/Southland_Basin_2.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Southland/Southland_outline_WGS84_2.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Southland/Southland_basement_WGS84.h5 basins/Southland_Basin_2.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

west_coast:
    @test -f basins/WestCoast.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WestCoast/WestCoast_basement_WGS84.h5 basins/WestCoast.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

te_anau:
    @test -f basins/TeAnau.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TeAnau/TeAnau_basement_WGS84.h5 basins/TeAnau.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

balclutha:
    @test -f basins/Balclutha.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Balclutha/Balclutha_basement_WGS84.h5 basins/Balclutha.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

castle_hill:
    @test -f basins/CastleHill.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h4 ${NZCVM_DATA_ROOT}/regional/CastleHill/CastleHill_basement_WGS84.h5 basins/CastleHill.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

cheviot:
    @test -f basins/Cheviot.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Cheviot/Cheviot_basement_WGS84.h5 basins/Cheviot.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

collingwood:
    @for i in 1 2 3; do \
        test -f basins/Collingwood${i}.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_outline_WGS84_${i}.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Collingwood/Collingwood_basement_WGS84.h5 basins/Collingwood${i}.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400; \
    done

dunedin:
    @test -f basins/Dunedin.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Dunedin/Dunedin_basement_WGS84.h5 basins/Dunedin.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

alexandra:
    @test -f basins/Alexandra.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Alexandra/Alexandra_basement_WGS84.h5 basins/Alexandra.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

gisborne:
    @test -f basins/Gisborne.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Gisborne/Gisborne_basement_WGS84.h5 basins/Gisborne.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

hakataramea:
    @test -f basins/Hakataramea.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Hakataramea/Hakataramea_basement_WGS84.h5 basins/Hakataramea.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

karamea:
    @test -f basins/Karamea.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Karamea/Karamea_basement_WGS84.h5 basins/Karamea.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

marlborough:
    @test -f basins/Marlborough.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Marlborough/Marlborough_basement_WGS84.h5 basins/Marlborough.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

mosgiel:
    @test -f basins/Mosgiel.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Mosgiel/Mosgiel_basement_WGS84.h5 basins/Mosgiel.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

murchison:
    @test -f basins/Murchison.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Murchison/Murchison_basement_WGS84.h5 basins/Murchison.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

ranfurly:
    @test -f basins/Ranfurly.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Ranfurly/Ranfurly_basement_WGS84.h5 basins/Ranfurly.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

rarakau:
    @test -f basins/Rarakau.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Rarakau/Rarakau_basement_WGS84.h5 basins/Rarakau.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

springs_junction:
    @test -f basins/SpringsJunction.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/SpringsJunction/SpringsJunction_basement_WGS84.h5 basins/SpringsJunction.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

tolaga_bay:
    @test -f basins/TolagaBay.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/TolagaBay/TolagaBay_basement_WGS84.h5 basins/TolagaBay.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waiapu:
    @test -f basins/Waiapu.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waiapu/Waiapu_basement_WGS84.h5 basins/Waiapu.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

waikato_hauraki:
    @test -f basins/WaikatoHauraki.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/WaikatoHauraki/WaikatoHauraki_basement_WGS84.h5 basins/WaikatoHauraki.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

wairarapa:
    @test -f basins/Wairarapa.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wairarapa/Wairarapa_basement_WGS84.h5 basins/Wairarapa.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 500

waitaki:
    @test -f basins/Waitaki.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Waitaki/Waitaki_basement_WGS84.h5 basins/Waitaki.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wakatipu:
    @test -f basins/Wakatipu.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wakatipu/Wakatipu_basement_WGS84.h5 basins/Wakatipu.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

wanaka:
    @test -f basins/Wanaka.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Wanaka/Wanaka_basement_WGS84.h5 basins/Wanaka.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

westport:
    @test -f basins/Westport.vtkhdf || {{ construct }} ${NZCVM_DATA_ROOT}/regional/Westport/Westport_outline_WGS84.geojson ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/surface/NZ_DEM_HD.h5 ${NZCVM_DATA_ROOT}/regional/Westport/Westport_basement_WGS84.h5 basins/Westport.vtkhdf --vm-1d ${NZCVM_DATA_ROOT}/vm1d/Cant1D_v2.fd_modfile -r 400

basins: canterbury hanmer mackenzie southland west_coast te_anau balclutha castle_hill cheviot collingwood dunedin alexandra gisborne hakataramea karamea marlborough mosgiel murchison ranfurly rarakau springs_junction tolaga_bay waiapu waikato_hauraki wairarapa waitaki wakatipu wanaka westport
