import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from argovisHelpers import helpers as avh
from Exploring_Argo_BGC_with_Argovis_helpers import polygon_lon_lat,  padlist, varrange, interpolate, \
    simple_map, plot_xycol, argo_heatmap, hurrplot, compare_plots

API_KEY='3e2bda40368d095888a54898b2f52c1fa50df102'
API_PREFIX = 'https://argovis-api.colorado.edu/'

dataDir = '/content/drive/MyDrive/SBS/data/'

def dfRead(dataSetName) :
    # Read dataframe from CSV file
    dfCSV = dataDir+dataSetName+'_dfm.csv'
    print(dfCSV)
    dfm = pd.read_csv(dfCSV)
    dfm.wmoid = dfm.wmoid.astype('category')

    dfmapCSV = dataSetName+'_dfmap.csv'
    dfmap = pd.read_csv(dfmapCSV)
    dfmap.wmoid = dfmap.wmoid.astype('category')

    # print(dfm.tail())

    return (dfm, dfmap)


def dfSave(dataSetName, dfm, dfmap) :

    # Save dataframe to CSV file
    dfCSV = dataDir+dataSetName+'_dfm.csv'
    print(dfCSV)
    dfm.to_csv(dfCSV)
    dfmapCSV = dataDir+dataSetName+'_dfmap.csv'
    dfmap.to_csv(dfmapCSV)



def getProfilesFromPolygon(polygon, startDate, endDate, platform_type ):

    nowiso = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    if endDate == '':
        endDate = nowiso+'Z'

    # Set up Argovis query parameters
    params = {
            'startDate': startDate,
            'endDate':   endDate,
            'source': 'argo_bgc',
            'polygon': eval(polygon),
            'data': 'cdom,salinity,temperature',
            'platform_type' : platform_type
        }
    print(params)

    # Make the query. Returns a list of JSON instances, one per profile
    dd = avh.query('argo', options=params, apikey=API_KEY, apiroot=API_PREFIX)

    # Create a dataframe from each profile JSON string (dfs is an array of dataframes)
    dfs = [pd.DataFrame.from_records([level for level in avh.data_inflate(profile)]) for profile in dd]

    # Pull out data from JSON data set that's not in the profile data
    # Used for plotting / forensics
    # dfs = [pd.DataFrame.from_records([level for level in avh.data_inflate(profile)]) for profile in dd]

    # Two accumulating data frames: dfm for profile data, dfmap for profile location
    dfm = pd.DataFrame([])
    dfmap = pd.DataFrame([])

    Nprof = len(dd)
    Ncdom = 0
    ptc = PlatformTypeCache()

    for profile,df in zip(dd, dfs):
        if 'cdom' in df.columns :
            Ncdom = Ncdom + 1

            dfp = pd.DataFrame([])
            platform_type = ptc.query(profile)
            dfp['salinity']=df['salinity']
            dfp['temperature']=df['temperature']
            dfp['cdom']=df['cdom']
            dfp['pressure']=df['pressure']
            dfp['id'] = profile['_id']
            dfp['wmoid'] = profile['_id'][0:7]
            dfp['platform_type'] = platform_type
            dfp['year'] = profile['timestamp'][0:4]
            dfm = pd.concat([dfm, dfp], axis=0)

            dfp = pd.DataFrame({'_id' : profile['_id'], \
                            'wmoid' : profile['_id'][0:7], \
                            'year'  : profile['timestamp'][0:4], \
                            'lon'   : profile['geolocation']['coordinates'][0], \
                            'lat'   : profile['geolocation']['coordinates'][1]}, index=[Ncdom-1])
            dfmap = pd.concat([dfmap, dfp])

    # Data from all profiles retrieved and in dataframe dfm.
    print(f'N profiles = {Nprof}')
    print(f'Ncdom profiles = {Ncdom}')
    return (dfm, dfmap)


def getProfilesFromFloats(platforms, startDate, endDate, doCorrection=False, CorrectionFactor=[], ScalingFactor=[]) :

    platforms_choke = []


    nowiso = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    if endDate == '':
        endDate = nowiso+'Z'


    dd = list()

    for platform in platforms :
        # print(platform)
        params = {
            'startDate': startDate,
            'endDate':   endDate,
            'platform': platform,
            'data': 'cdom,salinity,temperature',
            'source': 'argo_bgc'
        }
        print(params)
        d = avh.query('argo', options=params, apikey=API_KEY, apiroot=API_PREFIX)
        dd.extend(d)

    # Create a dataframe from each profile JSON string (dfs is an array of dataframes)
    dfs = [pd.DataFrame.from_records([level for level in avh.data_inflate(profile)]) for profile in dd]
        # dffs.extend(df)

    # print(dfs[0].head(5))
    # print(dfs[-1].head(5))

    #  Pull out data from JSON data set that's not in the profile data
    # Used for plotting / forensics
    # dfs = [pd.DataFrame.from_records([level for level in avh.data_inflate(profile)]) for profile in dd]

    # Two accumulating data frames: dfm for profile data, dfmap for profile location
    dfm = pd.DataFrame([])
    dfmap = pd.DataFrame([])

    Nprof = len(dd)
    Ncdom = 0
    ptc = PlatformTypeCache()

    for profile,df in zip(dd, dfs):
        if 'cdom' in df.columns :
            Ncdom = Ncdom + 1


            # print(Ncdom, wmoid, iwmo, CF)

            dfp = pd.DataFrame([])
            platform_type = ptc.query(profile)
            dfp['cdom']=df['cdom']
            dfp['salinity']=df['salinity']
            dfp['temperature']=df['temperature']

            if doCorrection :
                wmoid = int(profile['_id'][0:7])
                iwmo = np.where(platforms == wmoid)[0][0]
                CF = CorrectionFactor[iwmo]
                SF = ScalingFactor[iwmo]
                dfp['cdom_sc']  = dfp['cdom']*SF
                dfp['cdom_adj'] = dfp['cdom_sc']*CF
            dfp['pressure']=df['pressure']
            dfp['id'] = profile['_id']
            dfp['wmoid'] = profile['_id'][0:7]
            dfp['platform_type'] = platform_type
            dfp['year'] = profile['timestamp'][0:4]
            dfm = pd.concat([dfm, dfp], axis=0)

            dfp = pd.DataFrame({'_id' : profile['_id'], \
                            'wmoid' : profile['_id'][0:7], \
                            'year'  : profile['timestamp'][0:4], \
                            'lon'   : profile['geolocation']['coordinates'][0], \
                            'lat'   : profile['geolocation']['coordinates'][1], \
                            'platform_type' : platform_type}, index=[Ncdom-1])
            dfmap = pd.concat([dfmap, dfp])

    print(f'N profiles = {Nprof}')
    print(f'Ncdom profiles = {Ncdom}')
    return (dfm, dfmap)

def plotdfmByYear(dataSetName, dfm, var, xrange=[0,25]) :

    if len(dfm) > 700e3 :
        dfm = dfm.copy().iloc[::2]
        print(f'rows = {len(dfm)}, decimated')

    # Plot dfm, facety by year
    dfm.sort_values(by=['year'], inplace=True)

    fig = px.scatter(dfm, x=var, y='pressure', color='wmoid', facet_col='year', hover_data=['id'],  facet_col_wrap=7, title=f'{dataSetName} : {var}')
    fig.update_traces(marker={'size': 1.5})
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(width=1500, height=800, legend= {'itemsizing': 'constant'})
    fig.show()

    return

def plotdfmByYearAnim(dataSetName, dfm) :

    # Blindly decimate the dataframe
    if len(dfm) > 100e3 :
        dfm = dfm.copy().iloc[::16]
        print(f'rows = {len(dfm)}, decimated')

    # Rename the cdom_adj column to cdom_zadj so it sorts to the end 
    dfm.rename(columns={'cdom_adj': 'cdom_zadj'}, inplace=True)
    cols = ['cdom', 'cdom_sc', 'cdom_zadj']
    dfm2 = dfm.melt(id_vars=['wmoid', 'id', 'pressure', 'year'], value_vars = cols)
    dfm2.sort_values(by=['year', 'variable'], inplace=True)

    # Plot dfm2: color by wmoid, facet by year, animate by variable 
    fig = px.scatter(dfm2, x='value', y='pressure', animation_frame='variable', color='wmoid', facet_col='year', hover_data=['id'],  facet_col_wrap=7, title=f'{dataSetName}')
    fig.update_traces(marker={'size': 1.5})
    fig.update_xaxes(range=[0,20])
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(width=1500, height=800, legend= {'itemsizing': 'constant'})
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 5000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2000
    fig.show()

def plotdfmXYByYear(dataSetName, dfm, xvar, yvar) :

    if len(dfm) > 700e3 :
        dfm = dfm.copy().iloc[::2]
        print(f'rows = {len(dfm)}, decimated')
  
    # dfm.sort_values(by=['year'], inplace=True)
    # fig = px.scatter(dfm, x=xvar, y=yvar, color='wmoid', facet_col='year', hover_data=['id'],  facet_col_wrap=5, title=f'{dataSetName} : {var}')
    fig = px.scatter(dfm, x=xvar, y=yvar, color='wmoid',  hover_data=['id'], title=dataSetName)
    fig.update_traces(marker={'size': 1.5})
    fig.update_layout(width=1000, height=1000, legend= {'itemsizing': 'constant'})

    fig.show()

def cdom2000(dfm) :
    dfref = dfm[(dfm.pressure > 1900)]
    cdom2000ref = dfref.cdom_adj.mean(skipna=True)
    cdom2000std = dfref.cdom_adj.std(skipna=True)
    print(f'cdom2000ref={cdom2000ref}, std={cdom2000std}, CoV = {100*cdom2000std/cdom2000ref}%, refN={len(dfref.wmoid)}')


def getFloatMetadata(profile) :

    metaOptions = {
        'id': profile['metadata'][0]
    }
    md = avh.query('argo/meta', options=metaOptions, apikey=API_KEY, apiroot=API_PREFIX)
    # print(md[0])
    print(metaOptions, md[0]['platform_type'])
    return md[0]


@dataclass
class PlatformTypeCache() :

    pt_cache : Dict[str, str] = field(default_factory = dict)

    def query(self, d) -> str :

        platform_id = d['metadata'][0]

        # check cache
        if platform_id in self.pt_cache :
            platform_type = self.pt_cache[platform_id]  # return from cache
        else:
            md = getFloatMetadata(d)
            platform_type = md['platform_type']
            self.pt_cache[platform_id] = platform_type  # Add to cache
            # print(self.pt_cache)

        return platform_type



# ptc = PlatformTypeCache()
# ptc.query(d)
# ptc.query(d)



@dataclass
class CDOMcorr:
    df : pd.DataFrame = field(default_factory=pd.DataFrame)
    version : str = '20240508'
    dacdir : str = '/Users/ericrehm/Library/CloudStorage/OneDrive-Danaher/Sea-Bird/CDOM-Problem/FromMegan/'

    def __post_init__(self):
        dacpath = Path(self.dacdir).glob(f'*_CDOM_SN_aug_{self.version}.csv')
        self.df = pd.DataFrame([])
        for filename in dacpath :
            # print(filename.name)
            dfdac = pd.read_csv(filename, dtype={'SENSOR_SERIAL_NO':str})
            if not dfdac.empty :
                if self.df.empty :
                    self.df = dfdac 
                else:
                    self.df = pd.concat([self.df, dfdac],axis=0, ignore_index=True)
                # print(filename.name)
                # print(self.df.tail(1))
        
    def query(self, wmoid, var) :
        try :
            outvar = self.df[self.df["WMOID"] == wmoid][var].iloc[0]
            return outvar
        except :
            print(f'WMOID {wmoid} not found')
            return np.nan
    
    def gist(self) : 
        return self.df.head(3)

def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None,
        format: str='lonlat', projection: str='mercator',
        width_to_height: float=2.0):
    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460),
    ...     (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])/1.5

    if projection == 'mercator':
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )

    return zoom, center

def mapboxProfiles(dfmap, dataSetName, polygon='') :

    # Clean up float locations just for mapping
    dfmap = dfmap[(dfmap.lat != -90) & (dfmap.lon != 0)]
    dfmap.loc[:,'lon'] = dfmap.loc[:, 'lon'].apply(makeLonPostive)

    zoom, center = zoom_center(
        lons=dfmap.lon,
        lats=dfmap.lat
    )
    print(zoom, center)

    if not polygon:
        polygon = '[[]]'
    # dfmap.sort_values(by=['year'], inplace=True)
    # dfmap.sort_values('wmoid', inplace=True)

    try: 
        px.set_mapbox_access_token(open("/content/drive/MyDrive/Colab Notebooks/argovis/bgcargo").read())
    except:
        px.set_mapbox_access_token(open("/Users/ericrehm/.mapbox_token/bgcargo").read())

    fig = px.scatter_mapbox(dfmap, lat=dfmap.lat, lon=dfmap.lon, hover_name='_id', color='wmoid', opacity=1, size=np.ones(len(dfmap)), size_max=5)

    # fig.update_yaxes(title='latitude', visible=True, showticklabels=False)
    fig.update_geos(fitbounds="geojson", showframe=True)
    fig.update_layout(legend= {'itemsizing': 'constant'}, autosize=False, width=900, height=700,
        mapbox=dict(
            bearing=0,
            pitch=0,
            zoom=zoom,
            center=center,
            style="satellite",
            # style="carto-positron",
            layers=[{
                "below":'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source":["https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}"]
            },
            {
                'below': "traces",
                'source': {
                    'type': "FeatureCollection",
                    'features': [{
                        'type': "Feature",
                        'geometry': {
                            'type': "MultiPolygon",
                            'coordinates': [[literal_eval(polygon)]]
                        }
                    }]
                },
                'type': "line", 'color': "royalblue"
            }])
        )
    fig.show()

def makeLonPostive(x):
    if x < 0 == 0:
        return x + 360
    else:
        return x

def mapProfiles(dfmap, dataSetName, polygon='') :

    # Clean up float locations just for mapping
    dfmap = dfmap[(dfmap.lat != -90) & (dfmap.lon != 0)]
    dfmap.loc[:,'lon'] = dfmap.loc[:, 'lon'].apply(makeLonPostive)

    # Map locations, colored by WMOID
    # Note: simple_map was modified to use Seaborn scatterplot and also pass in marker
    if not polygon:
        simple_map(dfmap.lon, dfmap.lat, z=dfmap.wmoid, s=50, title=dataSetName)  #polygon=polygon, 
    else :
        simple_map(dfmap.lon, dfmap.lat, z=dfmap.wmoid, s=50, title=dataSetName, polygon=polygon)
    plt.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0, markerscale=2)

def geomapProfiles(dfmap, dataSetName, polygon=''):

    # Clean up float locations just for mapping
    dfmap = dfmap[(dfmap.lat != -90) & (dfmap.lon != 0)]
    dfmap.loc[:,'lon'] = dfmap.loc[:, 'lon'].apply(makeLonPostive)

    fig = px.scatter_geo(dfmap, lat=dfmap.lat, lon=dfmap.lon, hover_name='_id', color='wmoid', projection='equirectangular', opacity=0.7, title=dataSetName)
    if polygon :
        fig.add_traces(list(px.line_geo(lon=polygon_lon_lat(polygon)['lon'],lat=polygon_lon_lat(polygon)['lat']).update_traces(line_color='black', line_width=2).select_traces()))
    fig.update_yaxes(title='latitude', visible=True, showticklabels=True)
    fig.update_geos(fitbounds="locations", showframe=True, lataxis_showgrid=True, lonaxis_showgrid=True, lataxis_dtick=10.0, lonaxis_dtick=10.0,  showland=True, showcoastlines=True, showcountries=True, showlakes=True, showocean=False)
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.show()
