"""
🎵 Müzik DNA Analizörü - Flask Sunucu
Modelleri yükler, HTML'e tahmin sonuçları gönderir
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import json
import os

# ==================================================================
# UYGULAMAYI BAŞLAT
# ==================================================================
app = Flask(__name__)
CORS(app)

print("=" * 60)
print("🎵 MÜZİK DNA ANALİZÖRÜ - SUNUCU")
print("=" * 60)

# ==================================================================
# MODELLERİ YÜKLE
# ==================================================================
MODELS_DIR = "models"

print("\n📦 Modeller yükleniyor...")

# 1. K-Means modeli (kümeleme)
with open(os.path.join(MODELS_DIR, "kmeans.pkl"), "rb") as f:
    kmeans = pickle.load(f)
print(f"✅ K-Means yüklendi (k={kmeans.n_clusters})")

# 2. Scaler (Min-Max normalizasyon)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler yüklendi (K-Means için)")

# 3. SVM modeli (sınıflandırma - KAZANAN)
with open(os.path.join(MODELS_DIR, "svm.pkl"), "rb") as f:
    svm = pickle.load(f)
print("✅ SVM yüklendi (%97.8 accuracy)")

# 4. SVM için ayrı scaler
with open(os.path.join(MODELS_DIR, "scaler_clf.pkl"), "rb") as f:
    scaler_clf = pickle.load(f)
print("✅ Scaler (SVM için) yüklendi")

# 5. Küme isimleri
with open(os.path.join(MODELS_DIR, "cluster_names.json"), "r", encoding="utf-8") as f:
    cluster_names = json.load(f)
# JSON keys string olur, int'e çevir
cluster_names = {int(k): v for k, v in cluster_names.items()}
print(f"✅ Küme isimleri yüklendi ({len(cluster_names)} adet)")

# 6. Şarkı veritabanı
df = pd.read_csv(os.path.join(MODELS_DIR, "spotify_data.csv"))
print(f"✅ Veri yüklendi ({len(df):,} şarkı)")

# Akustik özellikler (modelin eğitildiği sıra ile)
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Benzerlik nedenleri için özellik metadata'sı
FEATURE_META = {
    'danceability':     {'label': 'Dans',         'icon': '💃'},
    'energy':           {'label': 'Enerji',       'icon': '⚡'},
    'loudness':         {'label': 'Ses',          'icon': '🔊'},
    'speechiness':      {'label': 'Konuşma',      'icon': '🎤'},
    'acousticness':     {'label': 'Akustik',      'icon': '🎸'},
    'instrumentalness': {'label': 'Enstrümantal', 'icon': '🎼'},
    'liveness':         {'label': 'Canlılık',     'icon': '🎪'},
    'valence':          {'label': 'Pozitiflik',   'icon': '😊'},
    'tempo':            {'label': 'Tempo',        'icon': '🥁'},
}

def compute_reasons(user_norm_row, song_norm_row, top_k=3):
    """
    Kullanıcı ile şarkının en yakın 'top_k' özelliğini döner.
    Skor: delta küçük + paylaşılan değer 0.5'ten uzak (karakterli) = öncelik.
    """
    scored = []
    for i, feat in enumerate(AUDIO_FEATURES):
        u = float(user_norm_row[i])
        s = float(song_norm_row[i])
        delta = abs(u - s)
        shared = (u + s) / 2.0
        # Karakteri olan özelliklere küçük bir bonus (skor ne kadar düşükse o kadar iyi)
        score = delta - 0.2 * abs(shared - 0.5)
        scored.append((feat, delta, shared, score))
    scored.sort(key=lambda x: x[3])

    reasons = []
    for feat, _, shared, _ in scored[:top_k]:
        meta = FEATURE_META[feat]
        if shared >= 0.66:
            qualifier = 'Yüksek'
        elif shared <= 0.33:
            qualifier = 'Düşük'
        else:
            qualifier = 'Orta'
        reasons.append({
            'feature': feat,
            'icon': meta['icon'],
            'label': f"{qualifier} {meta['label']}"
        })
    return reasons

print("\n🎉 Tüm modeller hazır!")


# ==================================================================
# API ENDPOINT'LERİ
# ==================================================================

@app.route('/api/info', methods=['GET'])
def api_info():
    """Genel bilgi döner"""
    return jsonify({
        'total_songs': len(df),
        'total_clusters': kmeans.n_clusters,
        'total_genres': df['track_genre'].nunique(),
        'cluster_names': cluster_names,
        'svm_accuracy': 0.978
    })


@app.route('/api/search', methods=['GET'])
def api_search():
    """Şarkı arama: /api/search?q=tarkan"""
    query = request.args.get('q', '').strip().lower()

    if len(query) < 2:
        return jsonify({'results': [], 'count': 0})

    # Filtrele
    mask = (
        df['track_name'].str.lower().str.contains(query, na=False) |
        df['artists'].str.lower().str.contains(query, na=False)
    )
    results = df[mask].head(50)

    # JSON'a çevir
    songs = []
    for _, row in results.iterrows():
        songs.append({
            'track_id': row['track_id'],
            'track_name': str(row['track_name']),
            'artists': str(row['artists']),
            'album_name': str(row['album_name']),
            'track_genre': str(row['track_genre']),
            'popularity': int(row['popularity']),
            'cluster': int(row['cluster']),
            'cluster_name': cluster_names[int(row['cluster'])],
            'danceability': float(row['danceability']),
            'energy': float(row['energy']),
            'valence': float(row['valence']),
            'acousticness': float(row['acousticness']),
            'instrumentalness': float(row['instrumentalness']),
            'speechiness': float(row['speechiness']),
            'liveness': float(row['liveness']),
            'loudness': float(row['loudness']),
            'tempo': float(row['tempo'])
        })

    return jsonify({'results': songs, 'count': len(songs)})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    🤖 GERÇEK MODEL TAHMİNİ
    HTML'den gelen 9 akustik değeri SVM ve K-Means ile değerlendir
    """
    data = request.get_json()

    # Akustik vektörü oluştur
    features = np.array([[
        data['danceability'],
        data['energy'],
        data['loudness'],
        data['speechiness'],
        data['acousticness'],
        data['instrumentalness'],
        data['liveness'],
        data['valence'],
        data['tempo']
    ]])

    # Normalizasyon (eğitimde kullandığımız scaler ile)
    features_norm = scaler.transform(features)
    features_norm_clf = scaler_clf.transform(features)

    # K-Means tahmini
    kmeans_pred = int(kmeans.predict(features_norm)[0])

    # SVM tahmini (KAZANAN model)
    svm_pred = int(svm.predict(features_norm_clf)[0])

    return jsonify({
        'kmeans_cluster': kmeans_pred,
        'kmeans_name': cluster_names[kmeans_pred],
        'svm_cluster': svm_pred,
        'svm_name': cluster_names[svm_pred],
        'models_agree': kmeans_pred == svm_pred
    })


print("✅ API endpoint'leri tanımlandı")

@app.route('/api/similar', methods=['POST'])
def api_similar():
    """
    🎵 BENZER ŞARKI ÖNERİSİ
    Verilen akustik değerlere en yakın N şarkıyı döner (varsayılan 10,
    request body'sindeki n_top/count/limit ile değiştirilebilir, max 200).
    """
    from sklearn.metrics.pairwise import euclidean_distances

    data = request.get_json()
    cluster_id = int(data['cluster'])
    n_top = int(data.get('n_top') or data.get('count') or data.get('limit') or 10)
    n_top = max(1, min(n_top, 200))

    # Kullanıcı vektörü
    user_features = np.array([[
        data['danceability'], data['energy'], data['loudness'],
        data['speechiness'], data['acousticness'], data['instrumentalness'],
        data['liveness'], data['valence'], data['tempo']
    ]])
    user_norm = scaler.transform(user_features)

    # Aynı kümedeki şarkıları al
    cluster_songs = df[df['cluster'] == cluster_id].copy()
    cluster_features = cluster_songs[AUDIO_FEATURES].values
    cluster_norm = scaler.transform(cluster_features)

    # Öklid mesafelerini hesapla
    distances = euclidean_distances(user_norm, cluster_norm)[0]
    cluster_songs['distance'] = distances

    # En yakın N'i al
    similar = cluster_songs.nsmallest(n_top, 'distance').reset_index(drop=True)
    max_dist = distances.max()

    # Top-10 için normalize edilmiş feature matrisini al (reasons hesabı için)
    similar_norm = scaler.transform(similar[AUDIO_FEATURES].values)

    # JSON'a çevir
    songs = []
    for i, row in similar.iterrows():
        match = int((1 - row['distance'] / max_dist) * 100) if max_dist > 0 else 100
        reasons = compute_reasons(user_norm[0], similar_norm[i], top_k=3)
        songs.append({
            'track_id': row['track_id'],
            'track_name': str(row['track_name']),
            'artists': str(row['artists']),
            'track_genre': str(row['track_genre']),
            'popularity': int(row['popularity']),
            'cluster': int(row['cluster']),
            'match_percent': match,
            'reasons': reasons
        })

    return jsonify({'songs': songs})


print("✅ Benzer şarkı endpoint'i eklendi")


@app.route('/api/cluster/<int:cluster_id>', methods=['GET'])
def api_cluster(cluster_id):
    """
    🎯 KÜMEDEKİ ŞARKILAR
    Verilen kümedeki şarkıları popülerliğe göre sıralı döner.
    """
    if cluster_id not in cluster_names:
        return jsonify({'error': 'Geçersiz küme'}), 404

    limit = int(request.args.get('limit', 50))
    cluster_songs = df[df['cluster'] == cluster_id].copy()
    total = len(cluster_songs)
    top = cluster_songs.nlargest(limit, 'popularity')

    songs = []
    for _, row in top.iterrows():
        songs.append({
            'track_id': row['track_id'],
            'track_name': str(row['track_name']),
            'artists': str(row['artists']),
            'album_name': str(row['album_name']),
            'track_genre': str(row['track_genre']),
            'popularity': int(row['popularity']),
            'cluster': int(row['cluster']),
            'cluster_name': cluster_names[int(row['cluster'])],
            'danceability': float(row['danceability']),
            'energy': float(row['energy']),
            'valence': float(row['valence']),
            'acousticness': float(row['acousticness']),
            'instrumentalness': float(row['instrumentalness']),
            'speechiness': float(row['speechiness']),
            'liveness': float(row['liveness']),
            'loudness': float(row['loudness']),
            'tempo': float(row['tempo'])
        })

    return jsonify({
        'cluster': cluster_id,
        'cluster_name': cluster_names[cluster_id],
        'total': total,
        'songs': songs
    })


@app.route('/api/cluster/<int:cluster_id>/top', methods=['GET'])
def api_cluster_top(cluster_id):
    """
    🌟 KÜME PROFİLİ
    Bir kümenin en popüler şarkılarını, en sık görülen sanatçılarını
    ve en yaygın türlerini döner. DNA sonuç sayfasında 'bu kümeyi tanı'
    bölümü için.
    """
    if cluster_id not in cluster_names:
        return jsonify({'error': 'Geçersiz küme'}), 404

    songs_n = max(1, min(50, int(request.args.get('songs', 12))))
    artists_n = max(1, min(30, int(request.args.get('artists', 8))))
    genres_n = max(1, min(20, int(request.args.get('genres', 6))))

    cs = df[df['cluster'] == cluster_id]
    total = len(cs)

    # Top songs by popularity
    top_songs_df = cs.nlargest(songs_n, 'popularity')
    top_songs = []
    for _, row in top_songs_df.iterrows():
        top_songs.append({
            'track_id': row['track_id'],
            'track_name': str(row['track_name']),
            'artists': str(row['artists']),
            'track_genre': str(row['track_genre']),
            'popularity': int(row['popularity']),
            'cluster': int(row['cluster']),
        })

    # Top artists: split multi-artist strings on ';' or ','
    # then count occurrences; rank by count, break ties by max popularity
    artist_count = {}
    artist_pop_sum = {}
    for _, row in cs.iterrows():
        raw = str(row['artists'])
        # split common delimiters
        parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
        if not parts:
            continue
        pop = int(row['popularity'])
        for a in parts:
            artist_count[a] = artist_count.get(a, 0) + 1
            artist_pop_sum[a] = artist_pop_sum.get(a, 0) + pop
    sorted_artists = sorted(
        artist_count.items(),
        key=lambda kv: (-kv[1], -(artist_pop_sum[kv[0]] / kv[1]))
    )[:artists_n]
    top_artists = [{
        'name': name,
        'count': int(cnt),
        'avg_popularity': round(artist_pop_sum[name] / cnt, 1),
    } for name, cnt in sorted_artists]

    # Top genres
    genre_count = cs['track_genre'].value_counts().head(genres_n)
    top_genres = [{'name': str(g), 'count': int(n)} for g, n in genre_count.items()]

    return jsonify({
        'cluster': cluster_id,
        'cluster_name': cluster_names[cluster_id],
        'total': int(total),
        'top_songs': top_songs,
        'top_artists': top_artists,
        'top_genres': top_genres,
    })


@app.route('/api/song/<track_id>', methods=['GET'])
def api_song(track_id):
    """
    🎵 TEKİL ŞARKI
    track_id ile tek bir şarkıyı tüm akustik değerleriyle döner.
    """
    match = df[df['track_id'] == track_id]
    if len(match) == 0:
        return jsonify({'error': 'Şarkı bulunamadı'}), 404

    row = match.iloc[0]
    return jsonify({
        'track_id': row['track_id'],
        'track_name': str(row['track_name']),
        'artists': str(row['artists']),
        'album_name': str(row['album_name']),
        'track_genre': str(row['track_genre']),
        'popularity': int(row['popularity']),
        'cluster': int(row['cluster']),
        'cluster_name': cluster_names[int(row['cluster'])],
        'danceability': float(row['danceability']),
        'energy': float(row['energy']),
        'valence': float(row['valence']),
        'acousticness': float(row['acousticness']),
        'instrumentalness': float(row['instrumentalness']),
        'speechiness': float(row['speechiness']),
        'liveness': float(row['liveness']),
        'loudness': float(row['loudness']),
        'tempo': float(row['tempo'])
    })


print("✅ Küme & tekil şarkı endpoint'leri eklendi")

# ==================================================================
# HTML DOSYASINI SERVIS ET
# ==================================================================
@app.route('/')
def serve_html():
    """Ana sayfa - index.html dosyasını döner"""
    return send_from_directory('.', 'index.html')


# ==================================================================
# SUNUCUYU BAŞLAT
# ==================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 SUNUCU BAŞLADI!")
    print("=" * 60)
    print("🌐 Tarayıcıda aç: http://localhost:5000")
    print("⛔ Durdurmak için: Ctrl+C")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, host='127.0.0.1')