import { initializeApp } from "https://www.gstatic.com/firebasejs/9.5.0/firebase-app.js";
import { getAuth, signInWithPopup, signOut, GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/9.5.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyBTBhINzhF6zxOW1qG5sE3rAorbIA-Irxs",
    authDomain: "genlabs-1b712.firebaseapp.com",
    projectId: "genlabs-1b712",
    storageBucket: "genlabs-1b712.appspot.com",
    messagingSenderId: "395602540360",
    appId: "1:395602540360:web:33e2e20b82c44433cba57d",
    databaseURL: ""
};

const app = initializeApp(firebaseConfig);

const provider = new GoogleAuthProvider();
const auth = getAuth();

export function GoogleLogin() {
    signInWithPopup(auth, provider)
        .then((result) => {
            // This gives you a Google Access Token. You can use it to access the Google API.
            const credential = GoogleAuthProvider.credentialFromResult(result);
            console.log(credential);

            $.ajax({
                url: "/authenticated",
                type: "POST",
                haders: {"Access-Control-Allow-Origin": "*"},
                success: function (res) {
                    document.cookie = "user=" + result.user.displayName
                    document.cookie = "email=" + result.user.email
                    document.cookie = "photo=" + result.user.photoURL
                    window.location.replace('authenticated')
                }
            });
        }).catch((error) => {
            console.log(error)
        });
}

export function GoogleLogout() {
    signOut(auth).then(() => {
        $.ajax({
            url: "/logout",
            type: "GET"
        })
    }).catch(error => console.log(error))
}