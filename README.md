# Analiza sentymentu recenzji filmowych

Aplikacja webowa pobiera recenzje filmów z oficjalnego TMDB API i analizuje ich tekst gotowym modelem `nlptown/bert-base-multilingual-uncased-sentiment`. Własny model nie jest trenowany. Model był dostrajany na recenzjach produktów i przewiduje ocenę tekstu w skali 1–5 gwiazdek.

## Uruchomienie

Wymagany jest Python 3.9 lub nowszy.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Utwórz bezpłatne konto TMDB i wygeneruj API Read Access Token w [ustawieniach API](https://www.themoviedb.org/settings/api). Aplikację można skonfigurować na jeden z trzech sposobów:

1. Wpisać token w polu aplikacji — jest to wygodne do lokalnego testu.
2. Ustawić zmienną środowiskową w PowerShell przed uruchomieniem:

   ```powershell
   $env:TMDB_API_READ_ACCESS_TOKEN = "wklej_tutaj_token"
   python -m streamlit run app.py
   ```

3. Przy wdrożeniu na Streamlit Community Cloud dodać sekret o nazwie `TMDB_API_READ_ACCESS_TOKEN` w ustawieniach aplikacji (**App settings → Secrets**):

   ```toml
   TMDB_API_READ_ACCESS_TOKEN = "wklej_tutaj_token"
   ```

Token ustawiony jako sekret lub zmienna środowiskowa nie będzie wymagany w interfejsie aplikacji. Dzięki wdrożeniu aplikacji z sekretem prowadzący może korzystać z udostępnionego adresu bez wpisywania własnego tokenu. Nie umieszczaj prawdziwego tokenu w kodzie, repozytorium ani w pliku przekazywanym prowadzącemu.

## Dane i model

- Recenzje do predykcji są pobierane z Internetu w trakcie działania aplikacji.
- Źródłem jest oficjalne [TMDB API](https://developer.themoviedb.org/).
- Model `nlptown/bert-base-multilingual-uncased-sentiment` był dostrajany na recenzjach w językach angielskim, niderlandzkim, niemieckim, francuskim, włoskim i hiszpańskim. Recenzje pobieramy w języku angielskim (`en-US`).
- Przewidywane 1–2 gwiazdki oznaczają sentyment negatywny, 3 neutralny, a 4–5 pozytywny.
- Kategorie sentymentu są wyświetlane w kolejności negatywny, neutralny, pozytywny; wykres używa odpowiednio czerwonego, żółtego i zielonego koloru, a tło aplikacji jest szare.
- Podsumowania sentymentu są pokazane w kolorowych kartach, a wycentrowany wykres słupkowy jest umieszczony w osobnym boksie poniżej kart. Średnia ocen autorów z pobranych recenzji jest wyświetlana obok tytułu filmu.
- Plakat wybranego filmu jest pobierany z TMDB Images i wyświetlany obok tytułu oraz podsumowania; jeśli go brakuje, pokazuje się komunikat zastępczy.
- Podsumowanie filmu zawiera reżysera (pobieranego z TMDB Credits) i krótki opis z wyników wyszukiwania, jeśli te dane są dostępne.
- Ocena wystawiona przez autora recenzji jest wyświetlana oddzielnie i nie wpływa na przewidywanie z tekstu.
- Model i jego tokenizator są pobierane z Hugging Face przy pierwszym uruchomieniu analizy, a następnie przechowywane w pamięci aplikacji. Pierwsze uruchomienie może potrwać dłużej; model działa lokalnie na CPU.
- Aplikacja nie zapisuje tokenu ani recenzji do trwałego magazynu.

## Ograniczenia

Nie każdy film ma recenzje w TMDB, a liczba wyników może być ograniczona. Model był trenowany na recenzjach produktów, nie filmów, więc może nie rozpoznawać poprawnie kontekstu kinowego, ironii i opinii mieszanych. Przewidywana ocena na podstawie tekstu może różnić się od oceny autora recenzji.

Aplikacja korzysta z TMDB API, ale nie jest wspierana ani certyfikowana przez TMDB. Przed publikacją lub dalszą dystrybucją należy zapoznać się z aktualnymi [warunkami TMDB](https://www.themoviedb.org/terms-of-use).
