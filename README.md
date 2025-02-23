# NeuWid_Abschlussprojekt

## Projektbeschreibung
Das folgende Projekt dient der Auswertung beliebiger ebener Mechanismen, welche vier Einschränkungen besitzen. 
    - Es werden nur ebene Mechanismen betrachtet
    - Nur Drehgelenke sind mit starren Gliedern verbunden. 
    - Nur ein Drehgelenk dient als Antrieb 
    - Ein Gelenk ist fest verankert
Es werden Winkel im Bereich von 0° bis 360° betrachtet.

## Installation
Um das Projekt als Code zu verwenden, müssen folgende Schritte durchgeführt werden:
1. Das [GitHub-Repository](https://github.com/np362/NeuWid_Abschlussprojekt) klonen:
2. Die benötigten Bibliotheken installieren:
   ```bash
   pip install -r requirements.txt
   ```
3. Das Projekt kann nun verwendet werden.

Um das Projekt über die Streamlit-App zu verwenden, müssen folgende Schritte durchgeführt werden:
1. Den Link zu unserer [Streamlit-App](https://neuwidproject.streamlit.app/) aufrufen: 
2. Die App kann nun verwendet werden.

## Anwendung
Der Mechanismus kann durch Eingabe von Punkten mit deren Koordinaten in eine Tabelle oder durch Laden eines bereits bestehendem Mechanismus erstellt werden. Der Mechanismus wird dann in einem Koordinatensystem dargestellt. Die Eingabe sowie die Ausgabe erfolgen für den Nutzer über die Streamlit-App.
Der Mechanismus wird hierbei nicht nur angezeigt, sondern kann auch gespeichert werden. Des Weiteren kann der Mechanismus auch animiert werden. Hierbei wird der Mechanismus in verschiedenen Stellungen dargestellt, sodass der Nutzer die Bewegung des Mechanismus nachvollziehen kann. Die Animation kann ebenfalls gespeichert werden. So auch die Bahnstrecke eines gewählten Punktes, sofern dieser nicht statisch ist.
