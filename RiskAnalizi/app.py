import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def tansiyon_coz(tansiyon_str):
    try:
        sistolik,diyastolik = tansiyon_str.split('/')
        return int(sistolik), int(diyastolik)
    except Exception as hata:
        return np.nan, np.nan
    
def main():
    veri = pd.read_excel('tansiyon.xlsx')
        
    veri[['sabah_sistolik','sabah_diyastolik']] = veri['Sabah Tansiyon'].apply(lambda x: pd.Series(tansiyon_coz(x)))   
    veri[['aksam_sistolik','aksam_diyastolik']] = veri['Akşam Tansiyon'].apply(lambda x: pd.Series(tansiyon_coz(x))) 

    features = veri[['sabah_sistolik','sabah_diyastolik','aksam_sistolik','aksam_diyastolik']].dropna()

    kmeans = KMeans(n_clusters=2, random_state=42)
    kumeler = kmeans.fit_predict(features)
    features['cluster'] = kumeler

    kume_ortalamaları = features.groupby('cluster').mean()[['sabah_sistolik','aksam_sistolik']]
    riskli_kume = kume_ortalamaları.mean(axis=1).idxmax()

    features['risk'] = features['cluster'].apply(lambda x: 'Risk' if x == riskli_kume else 'Normal')

    veri = veri.join(features['risk'])

    risk_sayisi = veri['risk'].value_counts().get('Risk',0)
    toplam = veri['risk'].count()
    genel_risk = 'Yüksek Tansiyon/ Kalp Riski' if risk_sayisi/toplam >= 0.5 else 'Normal Tansiyon'

    print('Günlük Risk Durumları')
    print(veri[['Tarih','risk']])
    print("\nGenel Değerlendirme:", genel_risk )

if __name__ == '__main__':
    main()      